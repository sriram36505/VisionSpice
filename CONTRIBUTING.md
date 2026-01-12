# Contributing to VisionSpice

Thank you for your interest in contributing to VisionSpice! We welcome contributions from researchers, students, and developers.

## Code of Conduct

Please be respectful and professional in all interactions with the community.

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported
- Create an issue with:
  - Clear title and description
  - Steps to reproduce
  - Expected vs actual behavior
  - Python and PyTorch versions

### Suggesting Enhancements

- Use a clear, descriptive title
- Provide a detailed description of the enhancement
- Explain why this enhancement would be useful
- List alternatives you've considered

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/VisionSpice.git
   cd VisionSpice
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Include type hints where possible
   - Write tests for new functionality

4. **Commit your changes**
   ```bash
   git commit -m "Concise description of changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Submit a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure all tests pass

## Coding Standards

- **Style**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Include type hints for better code clarity
- **Testing**: Write unit tests for new functionality
- **Comments**: Add comments for complex logic

## Development Setup

```bash
# Clone the repository
git clone https://github.com/sriram36505/VisionSpice.git
cd VisionSpice

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

## Running Tests

```bash
pytest tests/
```

## Code Quality

```bash
# Format code
black src/

# Check for style issues
flake8 src/

# Type checking
mypy src/
```

## Documentation

- Update relevant documentation when making changes
- Provide clear commit messages
- Reference issues and discussions appropriately

## Contribution Areas

We're particularly interested in contributions for:

- **Model Improvements**: Better architectures, attention mechanisms
- **Augmentation Techniques**: New data augmentation strategies
- **Performance Optimization**: Training and inference speedups
- **Documentation**: Tutorials, guides, and API documentation
- **Testing**: Unit and integration tests
- **Visualization Tools**: Improved result visualization and analysis

## Questions?

Feel free to:
- Open an issue for discussion
- Check existing discussions
- Review the documentation

Thank you for contributing to VisionSpice! 🚀
