# Contributing / Contribuindo

[English](#english) | [Portugues](#portugues)

---

## English

### How to Contribute

Thank you for your interest in contributing to ML Model Drift Monitor! Here are the guidelines to help you get started.

### Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/ml-model-drift-monitor.git
   cd ml-model-drift-monitor
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
4. Install dependencies:
   ```bash
   make install
   ```

### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run linter and formatter:
   ```bash
   make lint
   make format
   ```
4. Run tests:
   ```bash
   make test
   ```
5. Commit your changes following [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add new drift detection method"
   ```
6. Push and create a Pull Request

### Code Standards

- Python 3.11+ with type annotations
- Follow PEP 8 style guide (enforced by ruff)
- Write docstrings for all public functions and classes
- Maintain test coverage above 80%
- All tests must pass before merging

### Pull Request Guidelines

- Provide a clear description of the changes
- Reference related issues if applicable
- Include tests for new functionality
- Update documentation as needed

---

## Portugues

### Como Contribuir

Obrigado pelo seu interesse em contribuir com o ML Model Drift Monitor! Aqui estao as diretrizes para ajuda-lo a comecar.

### Primeiros Passos

1. Faca um fork do repositorio
2. Clone seu fork:
   ```bash
   git clone https://github.com/seu-usuario/ml-model-drift-monitor.git
   cd ml-model-drift-monitor
   ```
3. Crie um ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
4. Instale as dependencias:
   ```bash
   make install
   ```

### Fluxo de Desenvolvimento

1. Crie uma branch para sua feature:
   ```bash
   git checkout -b feature/nome-da-sua-feature
   ```
2. Faca suas alteracoes
3. Execute o linter e formatter:
   ```bash
   make lint
   make format
   ```
4. Execute os testes:
   ```bash
   make test
   ```
5. Faca commit seguindo [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: adicionar novo metodo de deteccao de drift"
   ```
6. Faca push e crie um Pull Request

### Padroes de Codigo

- Python 3.11+ com anotacoes de tipo
- Seguir o guia de estilo PEP 8 (aplicado pelo ruff)
- Escrever docstrings para todas as funcoes e classes publicas
- Manter cobertura de testes acima de 80%
- Todos os testes devem passar antes do merge

### Diretrizes para Pull Requests

- Fornecer uma descricao clara das alteracoes
- Referenciar issues relacionadas quando aplicavel
- Incluir testes para novas funcionalidades
- Atualizar a documentacao conforme necessario
