from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx
import json
import os
from datetime import date
from pathlib import Path
import re
from typing import Optional
from collections import OrderedDict

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
_pending_draft = None
_nutrition_cache = OrderedDict()
_NUTRITION_CACHE_MAX = 200

class LogRequest(BaseModel):
    text: str
    meal_type: str = "snacks"

class ChatRequest(BaseModel):
    text: str
    meal_type: str = "snacks"

class EditEntryRequest(BaseModel):
    text: str
    meal_type: str = "snacks"
    food_name: Optional[str] = None
    grams: Optional[float] = None
    calories: Optional[float] = None
    protein: Optional[float] = None
    carbs: Optional[float] = None
    fat: Optional[float] = None

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
    normalized = " ".join(text.strip().lower().split())
    model = cfg.get("aizolo_model", "gemini/gemini-2.5-flash")
    cache_key = f"{model}|{normalized}"
    if cache_key in _nutrition_cache:
        _nutrition_cache.move_to_end(cache_key)
        return dict(_nutrition_cache[cache_key])

    prompt = """Extract food and nutrition from user text.
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
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://chat.aizolo.com/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {cfg['aizolo_key']}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": text}
                ],
                "max_tokens": 180,
                "temperature": 0
            }
        )
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = content.replace("```json","").replace("```","").strip()
        parsed = json.loads(content)
        _nutrition_cache[cache_key] = parsed
        _nutrition_cache.move_to_end(cache_key)
        while len(_nutrition_cache) > _NUTRITION_CACHE_MAX:
            _nutrition_cache.popitem(last=False)
        return dict(parsed)

def apply_target_protein(text: str, nutrition: dict) -> dict:
    pattern = r"(?:for|to get|need)\s+(\d+(?:\.\d+)?)\s*g?\s*(?:of\s+)?protein"
    match = re.search(pattern, text.lower())
    if not match:
        return nutrition

    target = float(match.group(1))
    protein_per_100g = float(nutrition.get("protein_per_100g", 0))
    if protein_per_100g <= 0:
        return nutrition

    grams = round((target * 100) / protein_per_100g, 1)
    nutrition["grams"] = grams
    return nutrition

def parse_quick_add(text: str) -> Optional[dict]:
    t = text.strip().lower()
    has_cal = ("cal" in t) or ("kcal" in t)
    has_pro = "protein" in t or re.search(r"\bg\s*p\b", t)
    if not (has_cal and has_pro):
        return None

    def extract_number(patterns: list[str]) -> Optional[float]:
        for p in patterns:
            m = re.search(p, t)
            if m:
                return float(m.group(1))
        return None

    calories = extract_number([r"(\d+(?:\.\d+)?)\s*(?:k?cal)\b"])
    protein = extract_number([r"(\d+(?:\.\d+)?)\s*g?\s*protein\b", r"(\d+(?:\.\d+)?)\s*g\s*p\b"])
    carbs = extract_number([r"(\d+(?:\.\d+)?)\s*g?\s*carb(?:s)?\b", r"(\d+(?:\.\d+)?)\s*g\s*c\b"]) or 0.0
    fat = extract_number([r"(\d+(?:\.\d+)?)\s*g?\s*fat\b", r"(\d+(?:\.\d+)?)\s*g\s*f\b"]) or 0.0

    if calories is None or protein is None:
        return None

    name_match = re.search(r"(?:called|name)\s*[:=]?\s*([a-z0-9][a-z0-9 \-_]{1,40})$", t)
    food_name = name_match.group(1).strip() if name_match else "quickadd"

    # Use a fixed 100g serving so totals can map directly to per-100g fields.
    return {
        "food_name": food_name,
        "grams": 100.0,
        "calories_per_100g": float(calories),
        "protein_per_100g": float(protein),
        "carbs_per_100g": float(carbs),
        "fat_per_100g": float(fat),
    }

async def create_food_entry(n: dict, meal_type: str, cfg: dict) -> dict:
    g = n["grams"]
    c100 = n["calories_per_100g"]
    p100 = n["protein_per_100g"]
    cb100 = n["carbs_per_100g"]
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
                "meal_type": meal_type, "quantity": g,
                "unit": "g", "entry_date": str(date.today())
            }
        )
        if entry_resp.status_code not in (200, 201):
            raise HTTPException(500, f"Diary log failed: {entry_resp.text}")
        entry = entry_resp.json()

    return {
        "entry_id": entry.get("id"),
        "food_id": fid,
        "food_name": n["food_name"],
        "grams": g,
        "meal": meal_type,
        "macros": {
            "calories": round(c100 * g / 100, 1),
            "protein":  round(p100 * g / 100, 1),
            "carbs":    round(cb100 * g / 100, 1),
            "fat":      round(f100 * g / 100, 1)
        }
    }

def nutrition_from_manual(req: EditEntryRequest) -> Optional[dict]:
    manual_fields = [req.grams, req.calories, req.protein, req.carbs, req.fat]
    if not any(v is not None for v in manual_fields):
        return None
    if None in manual_fields:
        raise HTTPException(400, "Manual edit needs grams, calories, protein, carbs, and fat.")
    if req.grams is None or req.grams <= 0:
        raise HTTPException(400, "Grams must be greater than 0.")

    g = float(req.grams)
    return {
        "food_name": (req.food_name or req.text).strip() or "custom food",
        "grams": g,
        "calories_per_100g": round(float(req.calories) * 100 / g, 3),
        "protein_per_100g": round(float(req.protein) * 100 / g, 3),
        "carbs_per_100g": round(float(req.carbs) * 100 / g, 3),
        "fat_per_100g": round(float(req.fat) * 100 / g, 3),
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    global _pending_draft
    cfg = load_config()
    if not cfg.get("aizolo_key"):
        raise HTTPException(400, "Not configured — open Settings first")

    text = req.text.strip()
    lowered = text.lower()
    if lowered in {"log it", "yeah log it", "log this", "yep log it"}:
        if not _pending_draft:
            raise HTTPException(400, "No pending item. Ask me first, then say 'log it'.")
        created = await create_food_entry(_pending_draft["nutrition"], _pending_draft["meal_type"], cfg)
        _pending_draft = None
        return {
            "mode": "logged",
            "message": f"✅ {created['grams']}g {created['food_name']}",
            **created
        }

    quick_n = parse_quick_add(text)
    if quick_n is not None:
        n = quick_n
    else:
        try:
            n = await extract_nutrition(text, cfg)
            n = apply_target_protein(text, n)
        except Exception as e:
            raise HTTPException(400, f"Could not parse food: {e}")

    _pending_draft = {"nutrition": n, "meal_type": req.meal_type}
    grams = n["grams"]
    m = {
        "calories": round(n["calories_per_100g"] * grams / 100, 1),
        "protein": round(n["protein_per_100g"] * grams / 100, 1),
        "carbs": round(n["carbs_per_100g"] * grams / 100, 1),
        "fat": round(n["fat_per_100g"] * grams / 100, 1)
    }
    return {
        "mode": "preview",
        "message": f"{grams}g {n['food_name']}  ({m['protein']}g protein). Say 'log it' to save.",
        "food_name": n["food_name"],
        "grams": grams,
        "meal": req.meal_type,
        "macros": m
    }

@app.post("/log")
async def log_food(req: LogRequest):
    cfg = load_config()
    if not cfg.get("aizolo_key"):
        raise HTTPException(400, "Not configured — open Settings first")
    quick_n = parse_quick_add(req.text)
    if quick_n is not None:
        n = quick_n
    else:
        try:
            n = await extract_nutrition(req.text, cfg)
            n = apply_target_protein(req.text, n)
        except Exception as e:
            raise HTTPException(400, f"Could not parse food: {e}")

    created = await create_food_entry(n, req.meal_type, cfg)

    return {
        "success": True,
        "message": f"✅ {created['grams']}g {created['food_name']}",
        "entry_id": created["entry_id"],
        "food_name": created["food_name"],
        "grams": created["grams"],
        "meal": created["meal"],
        "macros": created["macros"]
    }

@app.delete("/entries/{entry_id}")
async def delete_entry(entry_id: str):
    cfg = load_config()
    cookie, _ = await get_session()
    headers = {"Cookie": cookie}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.delete(f"{cfg['sparky_url']}/api/food-entries/{entry_id}", headers=headers)
        if resp.status_code not in (200, 204):
            raise HTTPException(500, f"Delete failed: {resp.text}")
    return {"success": True}

@app.patch("/entries/{entry_id}")
async def edit_entry(entry_id: str, req: EditEntryRequest):
    cfg = load_config()
    if not cfg.get("aizolo_key"):
        raise HTTPException(400, "Not configured — open Settings first")
    manual_nutrition = nutrition_from_manual(req)
    if manual_nutrition is not None:
        n = manual_nutrition
    else:
        try:
            n = await extract_nutrition(req.text, cfg)
            n = apply_target_protein(req.text, n)
        except Exception as e:
            raise HTTPException(400, f"Could not parse edit: {e}")

    cookie, _ = await get_session()
    headers = {"Cookie": cookie}
    async with httpx.AsyncClient(timeout=30) as client:
        del_resp = await client.delete(f"{cfg['sparky_url']}/api/food-entries/{entry_id}", headers=headers)
        if del_resp.status_code not in (200, 204):
            raise HTTPException(500, f"Replace failed (delete step): {del_resp.text}")

    created = await create_food_entry(n, req.meal_type, cfg)
    return {"success": True, **created}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()

app.mount("/static", StaticFiles(directory="static"), name="static")
