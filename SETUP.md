# SETUP

> Getting and running this project.

## 01: Set Up Machine (Once Per Machine)

Follow the detailed instructions at:
[**01. Set Up Your Machine**](https://denisecase.github.io/pro-analytics-02/01-set-up-machine/)

## 02: Set Up Project (Once Per Project)

Fork this repo into your GitHub account.
In your repo Settings:

-  Go to Pages tab / Enable GitHub Pages / Build and deployment / set Source to **GitHub Actions**
-  Go to Advanced Security tab / Dependabot / Dependabot security updates / **Enable**
-  And Advanced Security tab / Dependabot / Grouped security updates / **Enable**

Open a machine terminal to the folder where you store your **Repos**, and run:

```shell
git clone https://github.com/YOURACCOUNT/train-100-unigram
cd train-100-unigram
code .
```

When VS Code opens, accept the Extension Recommendations (click **`Install All`** or similar when asked).

Use VS Code menu option `Terminal` / `New Terminal` to open a **VS Code terminal** in the root project folder.
Run the following commands:

```shell
uv self update
uv python pin 3.14
uv sync --extra dev --extra docs --upgrade
```

If asked: "We noticed a new environment has been created. Do you want to select it for the workspace folder?" Click **"Yes"**.

Install and run pre-commit checks (repeat git `add` and `commit` twice as needed):

```shell
uvx pre-commit install
git add -A
uvx pre-commit run --all-files
```

More detailed instructions are available at:
[**02. Set Up Your Project**](https://denisecase.github.io/pro-analytics-02/02-set-up-project/)

## 03: Daily Workflow (Working With Python Project Code)

In VS Code terminal, pull before starting work:

```shell
git pull
```

Run the Python source files:

```shell
uv run python src/toy_gpt_train/a_tokenizer.py
uv run python src/toy_gpt_train/b_vocab.py
uv run python src/toy_gpt_train/c_model.py
uv run python src/toy_gpt_train/d_train.py
uv run python src/toy_gpt_train/e_infer.py
```

Run Python checks and tests:

```shell
uv run ruff format .
uv run ruff check . --fix
uv run pytest

uv run pyright
uv run bandit -c pyproject.toml -r src
uv run validate-pyproject pyproject.toml
```

Save progress frequently (some tools may make changes; **re-run git `add` and `commit`**
as needed to get everything committed before pushing):

```shell
git add -A
git commit -m "update"
git push -u origin main
```

See detailed instructions and troubleshooting at:
[**03. Daily Workflow**](https://denisecase.github.io/pro-analytics-02/03-daily-workflow/)

## Resources

- [Pro-Analytics-02](https://denisecase.github.io/pro-analytics-02/) - guide to professional Python
