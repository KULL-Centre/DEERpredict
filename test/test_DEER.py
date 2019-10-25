#! /usr/bin/env python

import MDAnalysis
import time
import numpy as np

from DEERpredict.DEERPrediction import DEERPrediction

import logging
logger = logging.getLogger("MDAnalysis.app")


if __name__ == "__main__":
    MDAnalysis.start_logging()
    # load the reference protein structure
    try:
        proteinStructure = MDAnalysis.Universe('test_DEER.pdb')
        # proteinStructure = MDAnalysis.Universe('PR_3BVB.pdb', 'PR_3BVB_skipped.xtc')
    except:
        logger.critical("protein structure and/or trajectory not correctly specified")
        raise

    startTime = time.time()
    profile_analysis = DEERPrediction(proteinStructure, ['55', '55'], chains=['A', 'B'], replicas=1,
                                        output_file='test_DEER')
    logger.info("DONE with analysis, elapsed time %6i s" % (int(time.time() - startTime)))

    MDAnalysis.stop_logging()

    reference = np.loadtxt('test_DEER_reference.dat')
    testing = np.loadtxt('test_DEER-55A-55B.dat')

    np.testing.assert_array_equal(reference, testing)


def test_DEER():
    MDAnalysis.start_logging()
    # load the reference protein structure
    try:
        proteinStructure = MDAnalysis.Universe('test_DEER.pdb')
        # proteinStructure = MDAnalysis.Universe('PR_3BVB.pdb', 'PR_3BVB_skipped.xtc')
    except:
        logger.critical("protein structure and/or trajectory not correctly specified")
        raise

    startTime = time.time()
    profile_analysis = DEERPrediction(proteinStructure, ['55', '55'], chains=['A', 'B'], replicas=1,
                                        output_file='test_DEER')
    logger.info("DONE with analysis, elapsed time %6i s" % (int(time.time() - startTime)))

    MDAnalysis.stop_logging()

    reference = np.loadtxt('test_DEER_reference.dat')
    testing = np.loadtxt('test_DEER-55A-55B.dat')

    np.testing.assert_array_equal(reference, testing)
