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


def set_description(description, branch=None):
    r = _get_repo()
    if branch is None:
        branch = r.active_branch.name
    repo_writer = r.config_writer()
    repo_writer.set('branch "{}"'.format(branch), "description", description)
    repo_writer.release()


class Description(object):
    def set(self, description, branch=None):
        set_description(branch, description)

    def set_from_file(self, filename, branch=None):
        description = open(filename).readlines()
        set_description(description, branch=branch)


if __name__ == "__main__":
    fire.Fire({"list": list_branches, "description": Description})
