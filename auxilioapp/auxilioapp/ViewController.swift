//
//  ViewController.swift
//  auxilioapp
//
//  Created by Raymond Nucuta on 2/10/20.
//  Copyright © 2020 Raymond Nucuta. All rights reserved.
//

import UIKit
import SafariServices

class ViewController: UIViewController, SFSafariViewControllerDelegate {

    func openURL() {
        // check if website exists
        guard let url = URL(string: "https://auxilioapp.herokuapp.com") else {
            return
        }

        let safariVC = SFSafariViewController(url: url)
        present(safariVC, animated: true, completion: nil)
        safariVC.delegate = self
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
    }
    
    override func viewDidAppear(_ animated: Bool) {
        openURL()
    }
    
    func safariViewControllerDidFinish(_ controller: SFSafariViewController) {
        controller.dismiss(animated: true, completion: nil)
    }

}

