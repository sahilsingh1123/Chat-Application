import os
import chainlit as cl
from typing import Optional

# Data Layer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

# SQLAlchemy for users
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select
from sqlalchemy import Column, String, JSON
import uuid
import asyncio
from datetime import datetime

# Password hashing
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

# — Environment & Hashing —
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///chat.db")
SECRET = os.getenv("CHAINLIT_AUTH_SECRET")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# — SQLAlchemy Models & Session —
Base = declarative_base()


class UserModel(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    identifier = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    createdAt = Column(String, default=lambda: datetime.utcnow().isoformat())


async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


engine = create_async_engine(DATABASE_URL, echo=True)
asyncio.run(init_models())
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def add_user(username: str, password: str):
    hashed_password = pwd_ctx.hash(password)
    async with SessionLocal() as session:
        user = UserModel(identifier=username, password_hash=hashed_password)
        session.add(user)
        await session.commit()
        print(f"User '{username}' added successfully.")


# asyncio.run(add_user("sahil", "admin"))
# — Chainlit Data Layer —
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=DATABASE_URL)


# — Password Auth Callback —
@cl.password_auth_callback
async def auth_pw(username: str, password: str) -> Optional[cl.User]:
    async with SessionLocal() as session:
        result = await session.execute(
            select(UserModel).where(UserModel.identifier == username)
        )
        user = result.scalar_one_or_none()
        if user and pwd_ctx.verify(password, user.password_hash):
            return cl.User(identifier=username)
    return None


# — Chat Handlers —
@cl.on_chat_start
async def on_start():
    user = cl.user_session.get("user")
    await cl.Message(
        f"Welcome, {user.identifier}! Your history is loaded."
    ).send()


@cl.on_message
async def handle_message(msg: cl.Message):
    # Insert your LLM/response logic here…
    await cl.Message(f"You said: {msg.content}").send()
