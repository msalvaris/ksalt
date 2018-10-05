import fire
from git import Repo


def _get_repo():
    return Repo(search_parent_directories=True)


def _is_branch(section):
    return section.startswith("branch")


def list_branches():
    r = _get_repo()
    repo_reader = r.config_reader()
    sections = repo_reader.sections()
    for branch in filter(_is_branch, sections):
        print(repo_reader.items(branch))


def set_description(branch, description):
    r = _get_repo()
    repo_writer = r.config_writer()
    repo_writer.set("branch \"{}\"".format(branch), "description", description)
    repo_writer.release()


if __name__ == "__main__":
    fire.Fire({"list": list_branches, "description": set_description})
