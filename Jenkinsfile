pipeline {
    parameters {string(defaultValue: '', description: 'api-token', name: 'API_TOKEN')
	                  string(defaultValue: '', description: 'buildable phid', name: 'TARGET_PHID')
	                  string(defaultValue: 'docker_debian_testing', description: 'docker file to use', name: 'DOCKERFILE')
                      string(defaultValue: 'g++', description: 'c++ compiler', name: 'CXX_COMPILER')
      }

    agent {
        dockerfile {
            filename 'docker_debian_testing'
                     dir 'dockerfiles'
            }
    }

    environment {
      BUILD_DIR = 'build_docker_debian_testing'
                 OMPI_MCA_plm = 'isolated'
                 OMPI_MCA_btl = 'tcp,self'
    }
    options {
      disableConcurrentBuilds()
    }
    stages {
      stage ('wipe build') {
              when {
                  anyOf{
                      changeset glob: "**/*.cmake"
                                        changeset glob: "**/CMakeLists.txt"

                    }
                }
                   steps {
                     sh ' rm -rf ${BUILD_DIR}_${CXX_COMPILER}'
                   }
            }
      stage ('configure') {
              steps {
                  sh '''
                     mkdir -p ${BUILD_DIR}_${CXX_COMPILER}
                     cd ${BUILD_DIR}_${CXX_COMPILER}
                     CXX=${CXX_COMPILER} cmake -DCMAKE_BUILD_TYPE:STRING=Release -DRUNNING_IN_CI=ON ..
                     '''
                }
            }
            stage ('build') {
              steps {
                  sh 'make -C ${BUILD_DIR}_${CXX_COMPILER}'
                  }
              }


            stage ('Warnings gcc') {
              steps {
                  warnings(consoleParsers: [[parserName: 'GNU Make + GNU C Compiler (gcc)']])
                }
            }

            stage ('test') {
              steps {
                  sh '''
                     cd ${BUILD_DIR}_${CXX_COMPILER}
                     ctest || true
                     mkdir -p ../test_results
                     mv test_results*.xml ../test_results
                     '''
                  }
                  }
    }


    post {
      always {
          junit 'test_results/*.xml'
                sh 'rm -rf test_results/*'
                sh ''' set +x
                python3 -c "import os; import json; msg = {'buildTargetPHID':  os.environ['TARGET_PHID'],
                                                                'artifactKey': 'Jenkins URI {}'.format(os.environ['CXX_COMPILER']),
                                                                'artifactType': 'uri',
                                                                'artifactData': {
                                                                    'uri': os.environ['BUILD_URL'],
                                                                    'name': 'View Jenkins results',
                                                                    'ui.external': True
                                                                    }
                     }; print(json.dumps(msg))" | arc call-conduit --conduit-uri https://c4science.ch/ --conduit-token ${API_TOKEN} harbormaster.createartifact
                '''
        }

	         success {
               sh '''
                  set +x
                  python3 -c "import os; import json; msg = {'buildTargetPHID': os.environ['TARGET_PHID'], 'type':'pass'}; print(json.dumps(msg))" | arc call-conduit --conduit-uri https://c4science.ch/ --conduit-token ${API_TOKEN} harbormaster.sendmessage
                  '''
        }

	         failure {
               sh '''
                  set +x
                  python3 -c "import os; import json; msg = {'buildTargetPHID': os.environ['TARGET_PHID'], 'type':'fail'}; print(json.dumps(msg))" | arc call-conduit --conduit-uri https://c4science.ch/ --conduit-token ${API_TOKEN} harbormaster.sendmessage
                  '''
        }
    }
}
