# openFOAM, based on pyFoam

from os import path, system
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
import docker 

clients = docker.from_env()


def runFoam(command, container):
    container.exec_run('bash -c "source ./.bashrc; %s"' % (command))

def setupFoam(templateCase, nameFormat, updateConditions, container, rootFolder, **kwargs):
    case = templateCase.cloneCase(nameFormat)
    print('Processing %s' % nameFormat)
    # user specified
    updateConditions(case, templateCase, container, rootFolder, **kwargs)
    return path.split(case.name)[1]

def extractData(caseFolder, f):
    # find Force from postProcessing using criterion f
    forceFile = path.join(caseFolder, 'postProcessing', 'forces', '0', 'force.dat')
    fin = open(forceFile, "rt")
    data = fin.readlines()
    fin.close()
    # return data[-1]
    return f(data)