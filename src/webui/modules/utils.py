import os
import re
from datetime import datetime
from pathlib import Path

from src.webui.modules import github, shared
from src.webui.modules.logging_colors import logger


# Helper function to get multiple values from shared.gradio
def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [shared.gradio[k] for k in keys]


def save_file(fname, contents):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path_str = os.path.abspath(fname)
    rel_path_str = os.path.relpath(abs_path_str, root_folder)
    rel_path = Path(rel_path_str)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: \"{fname}\"')
        return

    with open(abs_path_str, 'w', encoding='utf-8') as f:
        f.write(contents)

    logger.info(f'Saved \"{abs_path_str}\".')


def delete_file(fname):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path_str = os.path.abspath(fname)
    rel_path_str = os.path.relpath(abs_path_str, root_folder)
    rel_path = Path(rel_path_str)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: \"{fname}\"')
        return

    if rel_path.exists():
        rel_path.unlink()
        logger.info(f'Deleted \"{fname}\".')


def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"


def atoi(text):
    return int(text) if text.isdigit() else text.lower()


# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_available_models():
    model_list = []
    for item in list(Path(f'{shared.args.model_dir}/').glob('*')):
        if not item.name.endswith(('.txt', '-np', '.pt', '.json', '.yaml', '.py')) and 'llama-tokenizer' not in item.name:
            model_list.append(item.name)

    return ['None'] + sorted(model_list, key=natural_keys)


def get_available_ggufs():
    model_list = []
    for item in Path(f'{shared.args.model_dir}/').glob('*'):
        if item.is_file() and item.name.lower().endswith(".gguf"):
            model_list.append(item.name)

    return ['None'] + sorted(model_list, key=natural_keys)


def get_available_presets():
    return sorted(set((k.stem for k in Path('presets').glob('*.yaml'))), key=natural_keys)


def get_available_prompts():
    prompt_files = list(Path('prompts').glob('*.txt'))
    sorted_files = sorted(prompt_files, key=lambda x: x.stat().st_mtime, reverse=True)
    prompts = [file.stem for file in sorted_files]
    prompts.append('None')
    return prompts


def get_available_characters():
    paths = (x for x in Path('characters').iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    return sorted(set((k.stem for k in paths)), key=natural_keys)
    # TODO: set up for use with existing personas


def get_available_instruction_templates():
    path = "instruction-templates"
    paths = []
    if os.path.exists(path):
        paths = (x for x in Path(path).iterdir() if x.suffix in ('.json', '.yaml', '.yml'))

    return ['None'] + sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_extensions():
    extensions = sorted(set(map(lambda x: x.parts[1], Path('extensions').glob('*/script.py'))), key=natural_keys)
    extensions = [v for v in extensions if v not in github.new_extensions]
    return extensions


def get_available_loras():
    return ['None'] + sorted([item.name for item in list(Path(shared.args.lora_dir).glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=natural_keys)


def get_datasets(path: str, ext: str):
    # include subdirectories for raw txt files to allow training from a subdirectory of txt files
    if ext == "txt":
        return ['None'] + sorted(set([k.stem for k in list(Path(path).glob('*.txt')) + list(Path(path).glob('*/')) if k.stem != 'put-trainer-datasets-here']), key=natural_keys)

    return ['None'] + sorted(set([k.stem for k in Path(path).glob(f'*.{ext}') if k.stem != 'put-trainer-datasets-here']), key=natural_keys)


from pathlib import Path


def get_project_root():
    # Assuming the script is one or two levels deep within the project directory
    return Path(__file__).parent.parent  # Adjust this based on your actual directory structure


def get_available_chat_styles():
    project_root = get_project_root()
    css_path = project_root / 'css'
    if not css_path.exists():
        raise FileNotFoundError(f"{css_path} does not exist.")

    return sorted(set(('-'.join(k.stem.split('-')[1:]) for k in css_path.glob('chat_style-*.css'))), key=natural_keys)


def get_available_grammars():
    return ['None'] + sorted([item.name for item in list(Path('grammars').glob('*.gbnf'))], key=natural_keys)
