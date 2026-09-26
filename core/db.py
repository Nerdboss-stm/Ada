"""One cached MongoClient for the cluster at MONGODB_URI (CONTRACTS.md §2)."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

ADA = "ada"
ADA_CKPT = "ada_ckpt"
ADA_CACHE = "ada_cache"
ADA_TEST = "ada_test"
DATABASES = (ADA, ADA_CKPT, ADA_CACHE, ADA_TEST)


@lru_cache(maxsize=1)
def client() -> MongoClient:
    load_dotenv()
    return MongoClient(os.environ["MONGODB_URI"], tz_aware=True)


def db(name: str = ADA) -> Database:
    if name not in DATABASES:
        raise ValueError(f"unknown database {name!r}; expected one of {DATABASES}")
    return client()[name]
