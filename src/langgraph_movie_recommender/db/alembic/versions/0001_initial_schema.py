"""Initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "library_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("tmdb_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("watched", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("review", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("media_type IN ('movie', 'tv')", name="ck_library_media_type"),
        sa.CheckConstraint("tmdb_id > 0", name="ck_library_tmdb_id"),
        sa.CheckConstraint(
            "rating IS NULL OR (rating BETWEEN 1 AND 10)",
            name="ck_library_rating",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("media_type", "tmdb_id", name="uq_library_media_tmdb"),
    )

    op.create_table(
        "preferences",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False, server_default="explicit"),
        sa.Column("source_title", sa.String(), nullable=True),
        sa.Column("source_media_type", sa.String(), nullable=True),
        sa.Column("source_tmdb_id", sa.Integer(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind IN ('like', 'dislike')", name="ck_preference_kind"),
        sa.CheckConstraint(
            "source IN ('explicit', 'inferred')",
            name="ck_preference_source",
        ),
        sa.CheckConstraint(
            "source_media_type IS NULL OR source_media_type IN ('movie', 'tv')",
            name="ck_preference_source_media_type",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("text", "kind", name="uq_preference_text_kind"),
    )


def downgrade() -> None:
    op.drop_table("preferences")
    op.drop_table("library_items")
