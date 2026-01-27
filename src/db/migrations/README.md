# Database Migrations

This directory is reserved for database migration files.

## Recommended Tools

- **Alembic** (already in requirements): SQLAlchemy's migration tool
- **Django-style migrations**: If you prefer that pattern

## Setup (when needed)

```bash
# Initialize Alembic
cd probabilistic-generative-model
alembic init src/db/migrations

# Create a migration
alembic revision --autogenerate -m "add_new_column"

# Apply migrations
alembic upgrade head
```

## Current State

The project currently uses `sql/schema.sql` for initial setup.

When the schema needs to evolve (new columns, tables, etc.):
1. Create an Alembic migration
2. Apply to all environments
3. Keep schema.sql updated for fresh installs

## Migration Best Practices

1. **Always backup** before applying migrations
2. **Test on staging** before production
3. **Keep migrations small** and focused
4. **Never edit** applied migrations
5. **Add data migrations** separately from schema migrations
