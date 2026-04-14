import time
import uuid
from typing import Optional, Literal

from sqlalchemy.orm import Session
from open_webui.internal.db import Base, get_db, get_db_context
from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, Float, Integer, String, Text

####################
# Memory DB Schema
####################

MemoryScope = Literal['personal', 'work', 'preference', 'general']

# Base importance weight per scope
SCOPE_WEIGHTS: dict[str, float] = {
    'personal': 1.0,
    'work': 0.9,
    'preference': 0.8,
    'general': 0.5,
}

# Regenerate the user profile every time this many new/updated facts accumulate
PROFILE_REGEN_INTERVAL = 10


class Memory(Base):
    __tablename__ = 'memory'

    id = Column(String, primary_key=True, unique=True)
    user_id = Column(String)
    content = Column(Text)
    scope = Column(String, default='general')
    source_date = Column(BigInteger, nullable=True)
    # Importance: 0.0–1.0, computed from scope weight × recency × access frequency
    importance_score = Column(Float, default=0.5)
    # How many times this memory was surfaced in context
    access_count = Column(Integer, default=0)
    last_accessed_at = Column(BigInteger, nullable=True)
    updated_at = Column(BigInteger)
    created_at = Column(BigInteger)


class MemoryModel(BaseModel):
    id: str
    user_id: str
    content: str
    scope: str = 'general'
    source_date: Optional[int] = None
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed_at: Optional[int] = None
    updated_at: int
    created_at: int

    model_config = ConfigDict(from_attributes=True)


####################
# Memory Profile
####################


class MemoryProfile(Base):
    """Compressed user profile, regenerated every PROFILE_REGEN_INTERVAL new facts."""
    __tablename__ = 'memory_profile'

    user_id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    fact_count_at_generation = Column(Integer, default=0)
    updated_at = Column(BigInteger, nullable=False)


class MemoryProfileModel(BaseModel):
    user_id: str
    content: str
    fact_count_at_generation: int = 0
    updated_at: int

    model_config = ConfigDict(from_attributes=True)


####################
# Forms
####################


class MemoriesTable:
    def insert_new_memory(
        self,
        user_id: str,
        content: str,
        scope: str = 'general',
        source_date: Optional[int] = None,
        db: Optional[Session] = None,
    ) -> Optional[MemoryModel]:
        with get_db_context(db) as db:
            id = str(uuid.uuid4())
            now = int(time.time())

            memory = MemoryModel(
                **{
                    'id': id,
                    'user_id': user_id,
                    'content': content,
                    'scope': scope,
                    'source_date': source_date or now,
                    'created_at': now,
                    'updated_at': now,
                }
            )
            result = Memory(**memory.model_dump())
            db.add(result)
            db.commit()
            db.refresh(result)
            if result:
                return MemoryModel.model_validate(result)
            else:
                return None

    def update_memory_by_id_and_user_id(
        self,
        id: str,
        user_id: str,
        content: str,
        scope: Optional[str] = None,
        source_date: Optional[int] = None,
        db: Optional[Session] = None,
    ) -> Optional[MemoryModel]:
        with get_db_context(db) as db:
            try:
                memory = db.get(Memory, id)
                if not memory or memory.user_id != user_id:
                    return None

                memory.content = content
                memory.updated_at = int(time.time())
                if scope is not None:
                    memory.scope = scope
                if source_date is not None:
                    memory.source_date = source_date

                db.commit()
                db.refresh(memory)
                return MemoryModel.model_validate(memory)
            except Exception:
                return None

    def get_memories(self, db: Optional[Session] = None) -> list[MemoryModel]:
        with get_db_context(db) as db:
            try:
                memories = db.query(Memory).all()
                return [MemoryModel.model_validate(memory) for memory in memories]
            except Exception:
                return []

    def get_memories_by_user_id(self, user_id: str, db: Optional[Session] = None) -> list[MemoryModel]:
        with get_db_context(db) as db:
            try:
                memories = db.query(Memory).filter_by(user_id=user_id).all()
                return [MemoryModel.model_validate(memory) for memory in memories]
            except Exception:
                return []

    def get_memory_by_id(self, id: str, db: Optional[Session] = None) -> Optional[MemoryModel]:
        with get_db_context(db) as db:
            try:
                memory = db.get(Memory, id)
                return MemoryModel.model_validate(memory)
            except Exception:
                return None

    def delete_memory_by_id(self, id: str, db: Optional[Session] = None) -> bool:
        with get_db_context(db) as db:
            try:
                db.query(Memory).filter_by(id=id).delete()
                db.commit()

                return True

            except Exception:
                return False

    def delete_memories_by_user_id(self, user_id: str, db: Optional[Session] = None) -> bool:
        with get_db_context(db) as db:
            try:
                db.query(Memory).filter_by(user_id=user_id).delete()
                db.commit()

                return True
            except Exception:
                return False

    def delete_memory_by_id_and_user_id(self, id: str, user_id: str, db: Optional[Session] = None) -> bool:
        with get_db_context(db) as db:
            try:
                memory = db.get(Memory, id)
                if not memory or memory.user_id != user_id:
                    return None

                db.delete(memory)
                db.commit()
                return True
            except Exception:
                return False

    def record_access(self, ids: list[str], db: Optional[Session] = None) -> None:
        """Increment access_count and update last_accessed_at for retrieved memories."""
        with get_db_context(db) as db:
            try:
                now = int(time.time())
                db.query(Memory).filter(Memory.id.in_(ids)).update(
                    {
                        Memory.access_count: Memory.access_count + 1,
                        Memory.last_accessed_at: now,
                    },
                    synchronize_session=False,
                )
                db.commit()
            except Exception:
                pass

    def update_importance_score(self, id: str, score: float, db: Optional[Session] = None) -> None:
        with get_db_context(db) as db:
            try:
                memory = db.get(Memory, id)
                if memory:
                    memory.importance_score = max(0.0, min(1.0, score))
                    db.commit()
            except Exception:
                pass

    def get_top_memories_by_importance(
        self, user_id: str, limit: int = 5, db: Optional[Session] = None
    ) -> list[MemoryModel]:
        with get_db_context(db) as db:
            try:
                memories = (
                    db.query(Memory)
                    .filter_by(user_id=user_id)
                    .order_by(Memory.importance_score.desc())
                    .limit(limit)
                    .all()
                )
                return [MemoryModel.model_validate(m) for m in memories]
            except Exception:
                return []

    def count_memories_by_user_id(self, user_id: str, db: Optional[Session] = None) -> int:
        with get_db_context(db) as db:
            try:
                return db.query(Memory).filter_by(user_id=user_id).count()
            except Exception:
                return 0


Memories = MemoriesTable()


####################
# Profile Table CRUD
####################


class MemoryProfileTable:
    def get_profile(self, user_id: str, db: Optional[Session] = None) -> Optional[MemoryProfileModel]:
        with get_db_context(db) as db:
            try:
                row = db.get(MemoryProfile, user_id)
                return MemoryProfileModel.model_validate(row) if row else None
            except Exception:
                return None

    def upsert_profile(
        self,
        user_id: str,
        content: str,
        fact_count: int,
        db: Optional[Session] = None,
    ) -> Optional[MemoryProfileModel]:
        with get_db_context(db) as db:
            try:
                row = db.get(MemoryProfile, user_id)
                now = int(time.time())
                if row:
                    row.content = content
                    row.fact_count_at_generation = fact_count
                    row.updated_at = now
                else:
                    row = MemoryProfile(
                        user_id=user_id,
                        content=content,
                        fact_count_at_generation=fact_count,
                        updated_at=now,
                    )
                    db.add(row)
                db.commit()
                db.refresh(row)
                return MemoryProfileModel.model_validate(row)
            except Exception:
                return None

    def delete_profile(self, user_id: str, db: Optional[Session] = None) -> bool:
        with get_db_context(db) as db:
            try:
                row = db.get(MemoryProfile, user_id)
                if row:
                    db.delete(row)
                    db.commit()
                return True
            except Exception:
                return False


MemoryProfiles = MemoryProfileTable()
