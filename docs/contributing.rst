************
Contributing
************

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
#. In the meantime there might have been changes on the master branch. So you need
   to merge these changes into your branch.

   #. ``git checkout master``
   #. ``git pull`` to get the latest changes.
   #. ``git checkout <name-for-your-branch>``
   #. ``git merge master``. This might lead to conflicts that you have to resolve
      manually.

#. Push your branch to github with ``git push origin <name-for-your-branch>``.
#. Go to github and switch to your branch.
#. Send a pull request from the web UI on github.
#. After you received comments on your code, you can simply update your
   pull request by pushing to the same branch again.
#. Once your changes are accepted, merge your branch into master. This can
   also be done by the last reviewer that accepts the pull request.

Git commit messages
-------------------
Have a look at this `guideline <https://github.com/erlang/otp/wiki/writing-good-commit-messages>`_.

Most important:

* Single line summary starting with a verb (50 characters)
* Longer summary if necessary (wrapped at 72 characters).

Editors like ``vim`` enforce these constraints automatically.


Style Guide
===========
Follow :pep:`8` styleguide. It is worth reading through the entire
styleguide, but the most importand points are summarized here.

Naming
------
* Functions and variables use ``snake_case``
* Classes use ``CamelCase``
* Constants use ``CAPITAL_SNAKE_CASE``

Spacing
-------
Spaces around infix operators and assignment

* ``a + b`` not ``a+b``
* ``a = 1`` not ``a=1``

An exception are keyword arguments

* ``some_function(arg1=a, arg2=b)`` not ``some_function(arg1 = a, arg2 = b)``

Use one space after separating commas

* ``some_list = [1, 2, 3]`` not ``some_list = [1,2,3]``

In general PyCharm's auto format (Ctrl + Alt + l) should be good enough.

Type annotation
---------------

Since Python 3.5 type annotation are supported.
They make sense for public interfaces, that should be kept consistent.

``def add(a: int, b: int) -> int:``

Docstrings
----------
Use `Google Style <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
for docstrings in everything that has a somewhat public interface.

Clean code
----------
And here our non exhaustive list to guidelines to write cleaner code.

#. Use meaningful variable names
#. Keep your code DRY (Don't repeat yourself) by abstracting into functions and classes.
#. Keep everything at the same level of abstraction
#. Functions without side effects
#. Functions should have a single responsibility
#. Be consistent, stick to conventions, use a styleguide
#. Use comments only for what cannot be described in code
#. Write comments with care, correct grammar and correct punctuation
#. Write tests if you write a module




`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_


Experiment workflow
===================

#. Define a hypothesis
#. Define set of parameters that is going to stay fixed
#. Define parameter to change (including possible values for the parameter)
#. Create a meaningful name for the experiment (group of experiment, name of parameter tested)
#. Make sure you set a seed (Pycharm: in run options append: "with seed=0")
#. Program experiment (set parameters) using our framework
#. Commit your changes locally to obtain commit hash: this is going to be logged by sacredboard
#. Make sure your experiment is logged to the database
#. Start the experiment
#. Interpret and document results in the docs: Please include Sacred- IDs of the runs
#. Push your local branch to github - to make all commits available to everyone


