import datetime
import logging
import os
import subprocess
import sys
from glob import glob
from subprocess import CalledProcessError, TimeoutExpired

import fire
from git import Repo
from model import model_path

logger = logging.getLogger(__name__)


def _single_submission(filenames):
    logger.info('Found single submission {}'.format(filenames[0]))
    return filenames[0]


def _multiple_submissions(filenames):
    logger.info('Found multiple submission')
    logger.info('Sorting submission chronologically')
    files = sorted(filenames, key=os.path.getctime)
    logger.info('Sorted Files {}'.format(files))
    logger.info('Selecting {}'.format(files[-1]))
    return files[-1]


def _no_submission_files(filename):
    raise FileNotFoundError('No submission files found')


_options = {0: _no_submission_files,
            1: _single_submission}


def _ver():
    repo = Repo()
    return repo.active_branch.name


def _submit(filename, dry_run=False):
    now = datetime.datetime.now()
    msg = 'Submitting {} from branch {} on {}'.format(filename, _ver(), str(now))
    logger.info(msg)
    try:
        if not dry_run:
            output = subprocess.run(['kaggle', 'competitions', 'submit',
                                     '-f', filename,
                                     '-m', msg,
                                     'tgs-salt-identification-challenge'],
                                    check=True, timeout=240, stdout=subprocess.PIPE)

        logger.info('Successful submission')
    except CalledProcessError:
        logger.warning('Submission Failed')
    except TimeoutExpired:
        logger.warning('Submission timed out')


def main(dry_run=False):
    submissions_path = model_path()
    filenames = glob(os.path.join(submissions_path, '*.csv'))
    subfile = _options.get(len(filenames), _multiple_submissions)(filenames)
    _submit(subfile, dry_run=dry_run)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(ch)
    fire.Fire(main)
