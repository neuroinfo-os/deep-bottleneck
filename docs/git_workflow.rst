Git workflow
============

This workflow describes the process of adding code to the repository.

#. Describe what you want to achieve in an issue.
#. Pull the master to get up to date.

    #. ``git checkout master``
    #. ``git pull``

#. Create a new local branch with ``git checkout -b <name-for-your-branch>``.
   It can make sense to prefix your branch with a description like ``feature`` or ``fix``.
#. Solve the issue, most probably in several commits.
#. Push your branch to github with ``git push origin <name-for-your-branch>``.
#. Go to github and switch to your branch.
#. Send a pull request from the web UI on github.
#. After you received comments on your code, you can simply update your
   pull request by pushing to the same branch again.
#. Once your changes are accepted, merge your branch into master. This can
   also be done by the last reviewer that accepts the pull request.
