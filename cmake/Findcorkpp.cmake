download_external_project(corkpp
  URL "https://github.com/afalsafi/cork.git"
  TAG "${_corkpp}"
  BACKEND GIT
  THIRD_PARTY_SRC_DIR ${_corkpp_external_dir}
  # ${_corkpp_update}
  NO_UPDATE
  )


add_subdirectory(${_corkpp_external_dir}/corkpp)
set (CORKPP_INCLUDE_DIR ${_corkpp_external_dir}/corkpp/src)
# add_subdirectory(${CORKPP_INCLUDE_DIR})
include_directories(SYSTEM ${CORKPP_INCLUDE_DIR})


