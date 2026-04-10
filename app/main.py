from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx
import json
import os
from datetime import date
from pathlib import Path

app = FastAPI()

CONFIG_PATH = Path("/data/config.json")
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}

def save_config(cfg: dict):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

_session = {"cookie": None, "user_id": None, "last_email": None}

class LogRequest(BaseModel):
    text: str
    meal_type: str = "snacks"

class ConfigPayload(BaseModel):
    sparky_url: str
    sparky_email: str
    sparky_pass: str
    aizolo_key: str
    aizolo_model: str = "gemini/gemini-2.5-flash"

@app.get("/config")
async def get_config():
    cfg = load_config()
    return {
        "sparky_url":   cfg.get("sparky_url",   "http://192.168.68.77"),
        "sparky_email": cfg.get("sparky_email",  ""),
        "sparky_pass":  "••••••••" if cfg.get("sparky_pass")  else "",
        "aizolo_key":   "••••••••" if cfg.get("aizolo_key")   else "",
        "aizolo_model": cfg.get("aizolo_model",  "gemini/gemini-2.5-flash"),
        "configured":   bool(cfg.get("sparky_email") and cfg.get("sparky_pass") and cfg.get("aizolo_key"))
    }

@app.post("/config")
async def set_config(payload: ConfigPayload):
    existing = load_config()
    cfg = {
        "sparky_url":   payload.sparky_url,
        "sparky_email": payload.sparky_email,
        "sparky_pass":  payload.sparky_pass if payload.sparky_pass != "••••••••" else existing.get("sparky_pass", ""),
        "aizolo_key":   payload.aizolo_key  if payload.aizolo_key  != "••••••••" else existing.get("aizolo_key", ""),
        "aizolo_model": payload.aizolo_model,
    }
    save_config(cfg)
    _session["cookie"] = None
    _session["user_id"] = None
    return {"ok": True}

async def get_session():
    cfg = load_config()
    email = cfg.get("sparky_email")
    if not email:
        raise HTTPException(400, "Not configured — open Settings first")
    if _session["cookie"] and _session["last_email"] == email:
        return _session["cookie"], _session["user_id"]
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{cfg['sparky_url']}/api/auth/sign-in/email",
            json={"email": email, "password": cfg["sparky_pass"]},
            headers={"Content-Type": "application/json"}
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(500, f"SparkyFitness login failed: {resp.text}")
        data = resp.json()
        user_id = data.get("user", {}).get("id")
        cookie_str = "; ".join(f"{k}={v}" for k, v in resp.cookies.items())
        _session["cookie"] = cookie_str
        _session["user_id"] = user_id
        _session["last_email"] = email
        return cookie_str, user_id

async def extract_nutrition(text: str, cfg: dict) -> dict:
    prompt = """Extract food and nutrition from the user message.
Always assume meat/poultry is RAW unless stated otherwise (grilled, cooked, baked etc).
If values are given per 100g, scale to actual weight.
If no values given, use standard raw nutrition values from your knowledge.
Return ONLY a JSON object, no markdown, no explanation:
{
  "food_name": "raw chicken breast",
  "grams": 100,
  "calories_per_100g": 120,
  "protein_per_100g": 22.5,
  "carbs_per_100g": 0,
  "fat_per_100g": 2.6
}"""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://chat.aizolo.com/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {cfg['aizolo_key']}", "Content-Type": "application/json"},
            json={
                "model": cfg.get("aizolo_model", "gemini/gemini-2.5-flash"),
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": text}
                ],
                "max_tokens": 300
            }
        )
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = content.replace("```json","").replace("```","").strip()
        return json.loads(content)

@app.post("/log")
async def log_food(req: LogRequest):
    cfg = load_config()
    if not cfg.get("aizolo_key"):
        raise HTTPException(400, "Not configured — open Settings first")
    try:
        n = await extract_nutrition(req.text, cfg)
    except Exception as e:
        raise HTTPException(400, f"Could not parse food: {e}")

    g    = n["grams"]
    c100 = n["calories_per_100g"]
    p100 = n["protein_per_100g"]
    cb100= n["carbs_per_100g"]
    f100 = n["fat_per_100g"]

    cookie, user_id = await get_session()
    headers = {"Cookie": cookie, "Content-Type": "application/json"}
    sparky_url = cfg["sparky_url"]

    async with httpx.AsyncClient(timeout=30) as client:
        food_resp = await client.post(
            f"{sparky_url}/api/foods",
            headers=headers,
            json={
                "name": n["food_name"], "brand": "", "user_id": user_id,
                "is_custom": True, "is_quick_food": True,
                "serving_size": 100, "serving_unit": "g",
                "calories": c100, "protein": p100, "carbs": cb100, "fat": f100,
                "saturated_fat": 0, "polyunsaturated_fat": 0, "monounsaturated_fat": 0,
                "trans_fat": 0, "cholesterol": 0, "sodium": 0, "potassium": 0,
                "dietary_fiber": 0, "sugars": 0, "vitamin_a": 0, "vitamin_c": 0,
                "calcium": 0, "iron": 0, "is_default": True,
                "glycemic_index": "None", "custom_nutrients": {}
            }
        )
        if food_resp.status_code not in (200, 201):
            if food_resp.status_code == 401:
                _session["cookie"] = None
            raise HTTPException(500, f"Food creation failed: {food_resp.text}")

        food = food_resp.json()
        fid  = food["id"]
        vid  = food["default_variant"]["id"]

        entry_resp = await client.post(
            f"{sparky_url}/api/food-entries",
            headers=headers,
            json={
                "food_id": fid, "variant_id": vid,
                "meal_type": req.meal_type, "quantity": g,
                "unit": "g", "entry_date": str(date.today())
            }
        )
        if entry_resp.status_code not in (200, 201):
            raise HTTPException(500, f"Diary log failed: {entry_resp.text}")

    return {
        "success": True,
        "message": f"✅ {g}g {n['food_name']}",
        "macros": {
            "calories": round(c100 * g / 100, 1),
            "protein":  round(p100 * g / 100, 1),
            "carbs":    round(cb100 * g / 100, 1),
            "fat":      round(f100 * g / 100, 1)
        }
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()

app.mount("/static", StaticFiles(directory="static"), name="static")
