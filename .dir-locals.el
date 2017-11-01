;;((nil . ((cmake-ide-build-dir . "build"))))
;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((nil . ((eval . (set (make-local-variable 'my-project-path)
                      (file-name-directory
                       (let ((d (dir-locals-find-file ".")))
                         (if (stringp d) d (car d))))))
         (eval . (setq cmake-ide-build-dir (concat my-project-path "build")))
         )))
