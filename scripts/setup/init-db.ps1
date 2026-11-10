Set-Location $PSScriptRoot\..\..\apps\api
python -c "from app.core.database import Base, engine; import app.models.entities; Base.metadata.create_all(bind=engine); print('database initialized')"
