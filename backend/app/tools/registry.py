"""Allowlisted local tools with validation and confirmation gates."""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Note, Task, ToolCall

ToolHandler = Callable[[AsyncSession, str, dict[str, Any]], Awaitable[dict[str, Any]]]

DESTRUCTIVE_TOOLS = {"delete_note", "complete_task", "delete_task"}

SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
SAFE_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in SAFE_BINOPS:
            return SAFE_BINOPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in SAFE_UNARY:
            return SAFE_UNARY[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"sqrt", "abs"}:
            if n.func.id == "sqrt":
                return math.sqrt(_eval(n.args[0]))
            return abs(_eval(n.args[0]))
        raise ValueError("Unsupported expression")

    return _eval(node)


async def create_note(db: AsyncSession, user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    note = Note(
        user_id=user_id,
        title=str(args.get("title") or "Untitled")[:256],
        content=str(args.get("content") or ""),
        tags=list(args.get("tags") or []),
    )
    db.add(note)
    await db.flush()
    return {"id": note.id, "title": note.title, "content": note.content}


async def read_notes(db: AsyncSession, user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    limit = min(int(args.get("limit", 20)), 100)
    result = await db.execute(
        select(Note).where(Note.user_id == user_id).order_by(Note.updated_at.desc()).limit(limit)
    )
    notes = result.scalars().all()
    return {
        "notes": [
            {"id": n.id, "title": n.title, "content": n.content, "tags": n.tags} for n in notes
        ]
    }


async def search_notes(db: AsyncSession, user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    q = str(args.get("query") or "").strip()
    if not q:
        return {"notes": []}
    result = await db.execute(
        select(Note)
        .where(
            Note.user_id == user_id,
            or_(Note.title.ilike(f"%{q}%"), Note.content.ilike(f"%{q}%")),
        )
        .order_by(Note.updated_at.desc())
        .limit(50)
    )
    notes = result.scalars().all()
    return {
        "notes": [
            {"id": n.id, "title": n.title, "content": n.content, "tags": n.tags} for n in notes
        ]
    }


async def delete_note(db: AsyncSession, user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    note_id = str(args.get("note_id") or "")
    note = await db.get(Note, note_id)
    if note is None or note.user_id != user_id:
        return {"ok": False, "error": "Note not found"}
    await db.delete(note)
    await db.flush()
    return {"ok": True, "deleted": note_id}


async def create_task(db: AsyncSession, user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    title = str(args.get("title") or "").strip()
    if not title:
        raise ValueError("Task title required")
    task = Task(
        user_id=user_id,
        title=title[:256],
        description=str(args.get("description") or ""),
        status="open",
    )
    db.add(task)
    await db.flush()
    return {"id": task.id, "title": task.title, "status": task.status}


async def list_tasks(db: AsyncSession, user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    status = args.get("status")
    stmt = select(Task).where(Task.user_id == user_id)
    if status:
        stmt = stmt.where(Task.status == status)
    stmt = stmt.order_by(Task.updated_at.desc()).limit(100)
    result = await db.execute(stmt)
    tasks = result.scalars().all()
    return {
        "tasks": [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "status": t.status,
            }
            for t in tasks
        ]
    }


async def complete_task(db: AsyncSession, user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    task_id = str(args.get("task_id") or "")
    task = await db.get(Task, task_id)
    if task is None or task.user_id != user_id:
        return {"ok": False, "error": "Task not found"}
    task.status = "completed"
    task.completed_at = datetime.now(timezone.utc)
    await db.flush()
    return {"ok": True, "id": task.id, "status": task.status}


async def delete_task(db: AsyncSession, user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    task_id = str(args.get("task_id") or "")
    task = await db.get(Task, task_id)
    if task is None or task.user_id != user_id:
        return {"ok": False, "error": "Task not found"}
    await db.delete(task)
    await db.flush()
    return {"ok": True, "deleted": task_id}


async def get_local_datetime(db: AsyncSession, user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": str(now.tzinfo),
    }


async def calculate(db: AsyncSession, user_id: str, args: dict[str, Any]) -> dict[str, Any]:
    expr = str(args.get("expression") or "").strip()
    if not expr or len(expr) > 200:
        raise ValueError("Invalid expression")
    value = _safe_eval(expr)
    return {"expression": expr, "result": value}


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "create_note": {
        "description": "Create a local note",
        "parameters": {"title": "string", "content": "string", "tags": "string[]"},
    },
    "read_notes": {
        "description": "List recent notes",
        "parameters": {"limit": "number"},
    },
    "search_notes": {
        "description": "Search notes by text",
        "parameters": {"query": "string"},
    },
    "delete_note": {
        "description": "Delete a note by id (requires confirmation)",
        "parameters": {"note_id": "string"},
        "destructive": True,
    },
    "create_task": {
        "description": "Create a task",
        "parameters": {"title": "string", "description": "string"},
    },
    "list_tasks": {
        "description": "List tasks",
        "parameters": {"status": "string"},
    },
    "complete_task": {
        "description": "Mark a task completed (requires confirmation)",
        "parameters": {"task_id": "string"},
        "destructive": True,
    },
    "delete_task": {
        "description": "Delete a task (requires confirmation)",
        "parameters": {"task_id": "string"},
        "destructive": True,
    },
    "get_local_datetime": {
        "description": "Get local date and time",
        "parameters": {},
    },
    "calculate": {
        "description": "Evaluate a simple arithmetic expression",
        "parameters": {"expression": "string"},
    },
}

HANDLERS: dict[str, ToolHandler] = {
    "create_note": create_note,
    "read_notes": read_notes,
    "search_notes": search_notes,
    "delete_note": delete_note,
    "create_task": create_task,
    "list_tasks": list_tasks,
    "complete_task": complete_task,
    "delete_task": delete_task,
    "get_local_datetime": get_local_datetime,
    "calculate": calculate,
}


class ToolExecutor:
    def __init__(self, db: AsyncSession, user_id: str, session_id: str | None = None) -> None:
        self.db = db
        self.user_id = user_id
        self.session_id = session_id

    def schemas(self) -> dict[str, Any]:
        return TOOL_SCHEMAS

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        if tool_name not in HANDLERS:
            raise PermissionError(f"Tool not allowlisted: {tool_name}")
        arguments = arguments or {}
        requires_confirmation = tool_name in DESTRUCTIVE_TOOLS
        if requires_confirmation and not confirmed:
            record = ToolCall(
                user_id=self.user_id,
                session_id=self.session_id,
                tool_name=tool_name,
                arguments=arguments,
                result={"pending_confirmation": True},
                status="pending_confirmation",
                requires_confirmation=True,
                confirmed=False,
            )
            self.db.add(record)
            await self.db.flush()
            return {
                "status": "pending_confirmation",
                "tool_call_id": record.id,
                "tool_name": tool_name,
                "arguments": arguments,
                "message": "Confirmation required before this action runs.",
            }

        try:
            result = await HANDLERS[tool_name](self.db, self.user_id, arguments)
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}
            status = "error"

        record = ToolCall(
            user_id=self.user_id,
            session_id=self.session_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            status=status,
            requires_confirmation=requires_confirmation,
            confirmed=confirmed if requires_confirmation else True,
        )
        self.db.add(record)
        await self.db.flush()
        return {"status": status, "tool_call_id": record.id, "result": result}
