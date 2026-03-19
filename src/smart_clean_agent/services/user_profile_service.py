import csv
from functools import lru_cache

from smart_clean_agent.utils.config_handler import agent_conf
from smart_clean_agent.utils.logger_handler import logger
from smart_clean_agent.utils.path_tool import get_abs_path

PROFILE_FIELDS = ["user_id", "city", "name", "house_type", "floor_type"]
REQUIRED_PROFILE_FIELDS = {"user_id", "city"}


@lru_cache(maxsize=1)
def _load_profiles_cached(csv_path: str) -> tuple[dict[str, str], ...]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not REQUIRED_PROFILE_FIELDS.issubset(fieldnames):
            raise ValueError("用户资料文件缺少必要字段")

        profiles: list[dict[str, str]] = []
        for row in reader:
            profile = {key: (value or "").strip() for key, value in row.items() if key}
            user_id = profile.get("user_id", "")
            city = profile.get("city", "")

            if not user_id or not city:
                logger.warning("[用户资料]存在缺失 user_id 或 city 的记录，已跳过")
                continue

            profiles.append(profile)

        return tuple(profiles)



def invalidate_user_profiles_cache() -> None:
    _load_profiles_cached.cache_clear()



def _normalize_profile(profile: dict[str, str]) -> dict[str, str]:
    normalized = {field: (profile.get(field, "") or "").strip() for field in PROFILE_FIELDS}
    if not normalized["user_id"] or not normalized["city"]:
        raise ValueError("用户资料必须包含 user_id 和 city")

    return normalized



def load_user_profiles(csv_path: str | None = None) -> list[dict[str, str]]:
    profile_path = csv_path or get_abs_path(agent_conf["user_profiles_path"])
    return [dict(profile) for profile in _load_profiles_cached(profile_path)]



def list_user_profiles(csv_path: str | None = None) -> list[dict[str, str]]:
    return load_user_profiles(csv_path)



def get_user_profile(user_id: str, csv_path: str | None = None) -> dict[str, str] | None:
    for profile in load_user_profiles(csv_path):
        if profile["user_id"] == user_id:
            return profile

    return None



def save_user_profiles(profiles: list[dict[str, str]], csv_path: str | None = None) -> None:
    profile_path = csv_path or get_abs_path(agent_conf["user_profiles_path"])
    normalized_profiles = [_normalize_profile(profile) for profile in profiles]

    with open(profile_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
        writer.writeheader()
        writer.writerows(normalized_profiles)

    invalidate_user_profiles_cache()



def upsert_user_profile(
    profile: dict[str, str],
    csv_path: str | None = None,
    original_user_id: str | None = None,
) -> dict[str, str]:
    profile_path = csv_path or get_abs_path(agent_conf["user_profiles_path"])
    normalized_profile = _normalize_profile(profile)
    profiles = load_user_profiles(profile_path)
    target_user_id = original_user_id or normalized_profile["user_id"]

    updated_profiles: list[dict[str, str]] = []
    found = False
    for current_profile in profiles:
        if current_profile["user_id"] == target_user_id:
            found = True
            if (
                normalized_profile["user_id"] != target_user_id
                and get_user_profile(normalized_profile["user_id"], profile_path) is not None
            ):
                raise ValueError("用户ID已存在")
            updated_profiles.append(normalized_profile)
            continue

        if current_profile["user_id"] == normalized_profile["user_id"] and target_user_id != normalized_profile["user_id"]:
            raise ValueError("用户ID已存在")

        updated_profiles.append(current_profile)

    if not found:
        if get_user_profile(normalized_profile["user_id"], profile_path) is not None:
            raise ValueError("用户ID已存在")
        updated_profiles.append(normalized_profile)

    save_user_profiles(updated_profiles, profile_path)
    return normalized_profile

