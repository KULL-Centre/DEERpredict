#! /usr/bin/env python

import MDAnalysis
import time
import numpy as np
from DEERpredict.FRETPrediction import FRETPrediction

import logging
logger = logging.getLogger("MDAnalysis.app")


# def test_FRET():
#     MDAnalysis.start_logging()
#     # load the reference protein structure
#     try:
#         proteinStructure = MDAnalysis.Universe('test_FRET.gro')
#     except:
#         logger.critical("protein structure and/or trajectory not correctly specified")
#         raise
#     startTime = time.time()
#     profile_analysis = FRETPrediction(proteinStructure, ['1', '12'], replicas=1,
#                                       libname_1='Alexa 594 50cutoff 3step',
#                                       libname_2='Alexa 488 50cutoff 3step', temperature=292,
#                                       output_file='test_FRET')
#     # print profile_analysis.results()

#     logger.info("DONE with analysis, elapsed time %6i s" % (int(time.time() - startTime)))

#     MDAnalysis.stop_logging()

#     reference = np.loadtxt('test_FRET_reference.dat')
#     testing = np.loadtxt('test_FRET-1-12.dat')

#     np.testing.assert_array_equal(reference, testing)


if __name__ == "__main__":
    #test_FRET()

    MDAnalysis.start_logging()
    # load the reference protein structure
    try:
        proteinStructure = MDAnalysis.Universe('test_FRET.gro')
    except:
        logger.critical("protein structure and/or trajectory not correctly specified")
        raise
    startTime = time.time()
    profile_analysis = FRETPrediction(proteinStructure, ['1', '12'], replicas=1,
                                      libname_1='Alexa 594 50cutoff 3step',
                                      libname_2='Alexa 488 50cutoff 3step', temperature=292,
                                      output_file='test_FRET')
    # print profile_analysis.results()

    logger.info("DONE with analysis, elapsed time %6i s" % (int(time.time() - startTime)))

    MDAnalysis.stop_logging()
