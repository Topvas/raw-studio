# RAW Studio — збірка .exe через GitHub Actions

GitHub Actions безкоштовно дає справжні Windows-машини: вони зберуть
`RAW_Studio.exe` (самодостатній, Python на цільовій машині НЕ потрібен).

## Шлях А — через веб-інтерфейс (найпростіше, без git, ~5 хвилин)

1. Увійдіть на https://github.com → натисніть **+** (праворуч зверху) →
   **New repository** → ім'я `raw-studio` → **Create repository**.
2. На сторінці нового репозиторію клацніть посилання
   **"uploading an existing file"**.
3. Перетягніть туди ДВА об'єкти з цього пакета: `raw_studio.py` і
   папку `.github` (перетягуйте папку целиком — GitHub приймає її разом
   із вкладеною workflows). Натисніть **Commit changes**.
4. Відкрийте вкладку **Actions** → збірка `build-windows-exe` почнеться
   сама. Чекайте ~4-6 хв: жовта крапка → зелена галочка.
5. Відкрийте виконання → унизу **Artifacts** → завантажте
   `RAW_Studio_windows` → всередині `RAW_Studio.exe`.
6. Подвійний клік на `RAW_Studio.exe` — готово.

Увага: артефакти зберігаються 30 днів. Щоб мати постійне посилання —
зробіть тег: у репозиторії **Releases → Draft a new release → виберіть
або створіть тег v1.0 → Publish** — воркфлоу прикріпить .exe до релізу.

## Шлях Б — через командний рядок (якщо стоїть git)

    git init && git add . && git commit -m "RAW Studio"
    gh repo create raw-studio --private --source . --push
    gh run watch            # спостерігати за збіркою
    gh run download -n RAW_Studio_windows

## Шлях В — локальна збірка без GitHub (опційно)

    py -m pip install numpy matplotlib pyinstaller
    py -m PyInstaller --onefile --windowed --name RAW_Studio raw_studio.py

Готовий файл: `dist\RAW_Studio.exe`.

## Примітки

- Перший запуск .exe триває ~5-10 с (self-розпакування в temp) — це
  нормально для --onefile.
- При бажанні додайте іконку: `--icon=app.ico` у кроці "Build exe".
- Перестворити .exe після правок raw_studio.py — просто зробіть новий
  Commit Changes (воркфлоу запуститься сам) або кнопку **Run workflow**.
