from setuptools import setup, find_packages

setup(
    name = 'svlearn-fine-tuning',
    version = '1.0',
    packages = find_packages(where='src'),
    package_dir = {'': 'src'},

    python_requires = '>=3.11',

    install_requires = [
        # DOCUMENTATION BUILDERS
            
        # mkdocstrings[python] -- Generates documentation from docstrings
        'mkdocstrings[python]',
        # mkdocs-material -- Packages mkdocs with several plugins
        'mkdocs-material',
        # mkdocs-material-extensions -- Adds additional features to mkdocs-material
        'mkdocs-material-extensions',
        # mkdocs-awesome-pages-plugins -- Allows Navigation to be generated from the filesystem
        'mkdocs-awesome-pages-plugin',
        # mknotebooks -- Allows Jupyter Notebooks to be included in the documentation
        'mknotebooks',
        # mkdocs-mermaid2-plugin -- Adds support for Mermaid diagrams
        'mkdocs-mermaid2-plugin',
        # mkdocs-include-markdown-plugin -- allows include content of different codes
        'mkdocs-include-markdown-plugin', 


        # Decorator | For creating decorators
        'decorator',

        # Pathlib | For handling file paths
        'pathlib',

        # PyKwalify | For validating YAML files
        'pykwalify',

        # Pytest | For testing
        'pytest',

        # Rich | Rich Console Output
        'rich',

        # A great library for logging
        'loguru',

       # for configuration
        'ruamel.yaml',
       
       # for pydantic and instructor
        'pydantic', 'instructor',
         
       # To convert markdown to html
        'markdown',
        
        # For some core dependencies in all projects.
        'numpy==1.26.4','pandas','scikit-learn', 'seaborn', 'altair','plotly', 'matplotlib',
        
        # Jupyter needed for jupyter notebooks
        'jupyter',

        # Needed for bert_encoders
        'torch', 'transformers==4.45.2', 'datasets', 'tf-keras', 'peft', 

        # Needed for sentence-bert encoders: sbert_encoder_fine_tuning.ipynb
        'sentence_transformers',
        
        # for time-series (also will need pip install chronos from the github link as mentioned in time series lab notes)
        'pyarrow',

        # Needed for diffusion model
        'diffusers', 'pillow', 'torchvision',
        
        # Needed for full finetuning of roberta base classifier
        'evaluate',

])
