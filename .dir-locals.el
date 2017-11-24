;;((nil . ((cmake-ide-build-dir . "build"))))
;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((nil . ((eval .
               (setq cmake-ide-build-dir
                     (concat
                      (locate-dominating-file
                       default-directory
                       dir-locals-file)
                      "build"))))))
