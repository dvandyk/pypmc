# This Makefile implements common tasks needed by developers
# A list of implemented rules can be obtained by the command "make help"

.DEFAULT_GOAL=build
.PHONY : help
help :
	@echo ; \
	echo "    Implemented targets:"; \
	echo ; \
	echo "    build        build pypmc for python2 and python3"; \
	echo "    buildX       build pypmc for pythonX only where X is one of {2,3}"; \
	echo "    build-sdist  build pypmc from the dist directory"; \
	echo "    check        use nosetests to test pypmc with python 2.7 and 3"; \
	echo "    checkX       use nosetests to test pypmc with pythonX,";  \
	echo "                 where X is one of {2.7,3}"; \
	echo "    check-fast   use nosetests to run only quick tests of pypmc"; \
	echo "                 using nosetests-2.7 and nosetests3"; \
	echo "    check-sdist  use nosetests test the distribution generated by"; \
	echo "                 'make sdist'"; \
	echo "    clean        delete compiled and temporary files"; \
	echo "    coverage     produce and show a code coverage report" ; \
	echo "                 Note: Cython modules cannot be analyzed" ; \
	echo "    distcheck    runs 'check', check-sdist', 'run-examples' and"; \
	echo "                 opens a browser with the built documentation"; \
	echo "    doc          build the documetation using shpinx"; \
	echo "    help         show this message"; \
	echo "    run-examples run all examples using python 2 and 3" ; \
	echo "    sdist        make a source distribution" ; \
	echo "    show-todos   show todo marks in the source code" ; \
	echo

.PHONY : clean
clean:
	#remove build doc
	rm -rf ./doc/_build

	#remove .pyc files created by python 2.7
	rm -f ./*.pyc
	find -P . -name '*.pyc' -delete

	#remove .pyc files crated by python 3
	rm -rf ./__pycache__
	find -P . -name __pycache__ -delete

	#remove build folder in root directory
	rm -rf ./build

	#remove cythonized C source and object files
	find -P . -name '*.c' -delete

	#remove variational binaries only if command line argument specified
	find -P . -name '*.so' -delete

	#remove backup files
	find -P . -name '*~' -delete

	#remove files created by coverage
	rm -f .coverage
	rm -rf coverage

	# remove egg info
	rm -rf pypmc.egg-info

	# remove downloaded seutptools
	rm -f setuptools-3.3.zip

	# remove dist/
	rm -rf dist

.PHONY : build
build : build2 build3

.PHONY : build2
build2 :
	@# type python2 tests if the command python2 can be invoked
	if type python2 ; then python2 setup.py build_ext --inplace ; fi
.PHONY : build3
build3 :
	@# type python3 tests if the command python3 can be invoked
	if type python3 ; then python3 setup.py build_ext --inplace ; fi

.PHONY :
check : check2 check3

.PHONY : check2
check2 : build
	# type nosetestsX tests if the command nosetestsX can be invoked
	if type nosetests-2.7 ; then nosetests-2.7 . --processes=-1 --process-timeout=60 ; fi
	# run parallel tests only if mpi4py available
	if python2 -c "import mpi4py" &>/dev/null ; then \
	    mpirun -n 2 nosetests-2.7 . ; \
	fi

.PHONY : check3
check3 : build
	# type nosetestsX tests if the command nosetestsX can be invoked
	type nosetests3 &>/dev/null && nosetests3 . --processes=-1 --process-timeout=60
	# run parallel tests only if mpi4py available
	if python3 -c "import mpi4py" &>/dev/null ; then \
	    mpirun -n 2 nosetests3 . ; \
	fi

.PHONY : check-fast
check-fast : build
	if type nosetests-2.7 ; then nosetests-2.7  pypmc -a '!slow' --processes=-1 --process-timeout=60 ; fi
	if type nosetests3 ; then nosetests3 . -a '!slow' --processes=-1 --process-timeout=60 ; fi

.PHONY : .build-system-default
.build-system-default :
	python setup.py build_ext --inplace

.PHONY : doc
doc : .build-system-default
	cd doc; make html

.PHONY : run-examples
run-examples : build
	cd examples ; \
	for file in $$(ls) ; \
	do \
	    echo running $${file}; \
	    python2 $${file} ; \
	    python3 $${file} ; \
	done ; \
	# run mpi examples in parallel \
	echo parallel pmc ; \
	mpirun -n 10 python2 pmc_mpi.py ; \
	mpirun -n 10 python3 pmc_mpi.py

.PHONY : sdist
sdist :
	python setup.py sdist

.PHONY : build-sdist
build-sdist : sdist
	cd dist ; \
	tar xaf * ; \
	cd * ; \
	python2 setup.py build ; \
	python3 setup.py build

.PHONY : check-sdist
check-sdist : build-sdist
	cd dist/*/build ; \
	cd lib*2.7 ; nosetests-2.7 . --processes=-1 --process-timeout=60 ; \
	mpirun -n 2 nosetests-2.7  ; \
	cd ../lib*3.* ; nosetests3 . --processes=-1 --process-timeout=60 ; \
	mpirun -n 2 nosetests3

.PHONY : distcheck
distcheck : check check-sdist doc run-examples
	xdg-open link_to_documentation

.PHONY : show-todos
grep_cmd = 	grep -i -R -G --color=auto [^"au""sphinx.ext."]todo
show-todos :
	$(grep_cmd) pypmc
	$(grep_cmd) examples

.PHONY : coverage
coverage : .build-system-default
	rm -rf coverage
	nosetests  . --with-coverage --cover-package=pypmc --cover-html --cover-html-dir=coverage
	xdg-open coverage/index.html
