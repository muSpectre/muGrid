pipeline {
    parameters {string(defaultValue: '', description: 'api-token', name: 'API_TOKEN')
	                  string(defaultValue: '', description: 'buildable phid', name: 'TARGET_PHID')
	                  string(defaultValue: 'docker_debian_testing', description: 'docker file to use', name: 'DOCKERFILE')
                      string(defaultValue: 'g++', description: 'c++ compiler', name: 'CXX_COMPILER')
      }

    agent {
        dockerfile {
            filename '${DOCKERFILE}'
                     dir 'dockerfiles'
            }
    }

    environment {
      BUILD_DIR = 'build_${DOCKERFILE}'
                 OMPI_MCA_plm = 'isolated'
                 OMPI_MCA_btl = 'tcp,self'
    }

    stages {
      stage ('configure') {
              steps {
                  sh '''
                     mkdir -p ${BUILD_DIR}
                     cd ${BUILD_DIR}
                     CXX=${CXX_COMPILER} cmake -DCMAKE_BUILD_TYPE:STRING=Release ..
                     '''
                }
            }
            stage ('build') {
              steps {
                  sh 'make -C ${BUILD_DIR}'
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
                     cd ${BUILD_DIR}
                     ctest
                     '''
                  }
                  }
    }


    post {
      always {
          junit 'test_results*.xml'
        }

	         success {
               sh '''
                  set +x
                  echo "{
                    \"buildTargetPHID\": \"${TARGET_PHID}\",
                      \"type\": \"pass\"
                      }" | arc call-conduit --conduit-uri https://c4science.ch/ --conduit-token ${API_TOKEN} harbormaster.sendmessage
                  '''
        }

	         failure {
               sh '''
                  set +x
                  echo "{
                    \"buildTargetPHID\": \"${TARGET_PHID}\",
                     \"type\": \"fail\"
                     }" | arc call-conduit --conduit-uri https://c4science.ch/ --conduit-token ${API_TOKEN} harbormaster.sendmessage
                  '''
        }
    }
}
