//
//  ViewController.swift
//  PoseDetector
//

//

import UIKit


class ViewController: UIViewController {

    @IBOutlet weak var buttonsStackView: UIStackView!
    
    @IBOutlet weak var imageButton1: UIButton!
    
    @IBOutlet weak var imageButton2: UIButton!
    
    @IBOutlet weak var imageButton3: UIButton!
    
    @IBOutlet weak var imageButton4: UIButton!
    
    @IBOutlet weak var imageButton5: UIButton!
    
    @IBOutlet weak var imageButton6: UIButton!
    
    @IBOutlet weak var imageButton7: UIButton!
    
    
    var imageButtons: [UIButton] = []
    
    @IBOutlet weak var imageContainerView: ImageContainerView!
    
    @IBOutlet weak var imageView: UIImageView!
    
    var blurView1: UIVisualEffectView!
    var blurView2: UIVisualEffectView!
    var blurView3: UIVisualEffectView!
    var blurView4: UIVisualEffectView!

    let dataManager = DataManager.shared
    
    override func viewDidLoad() {
        super.viewDidLoad()

        setupUI()
    }

    func setupUI() {
        
        let beffect1 = UIBlurEffect(style: .light)
        blurView1 = UIVisualEffectView(effect: beffect1)
        
        let beffect2 = UIBlurEffect(style: .dark)
        blurView2 = UIVisualEffectView(effect: beffect2)

        let beffect3 = UIBlurEffect(style: .light)
        blurView3 = UIVisualEffectView(effect: beffect3)

        let beffect4 = UIBlurEffect(style: .dark)
        blurView4 = UIVisualEffectView(effect: beffect4)

        
        imageContainerView.addSubview(blurView1)
        imageContainerView.addSubview(blurView2)
        imageContainerView.addSubview(blurView3)
        imageContainerView.addSubview(blurView4)
        imageContainerView.bringSubviewToFront(blurView1)
        imageContainerView.bringSubviewToFront(blurView2)
        imageContainerView.bringSubviewToFront(blurView3)
        imageContainerView.bringSubviewToFront(blurView4)

        imageButtons = [imageButton1, imageButton2, imageButton3, imageButton4, imageButton5, imageButton6, imageButton7]
        
        for (index, button) in imageButtons.enumerated() {

            button.tag = index
        }
        

        onClickedImageButton(imageButton1)
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        blurView1.frame = imageContainerView.bounds
        blurView2.frame = imageContainerView.bounds
        blurView3.frame = imageContainerView.bounds
        blurView4.frame = imageContainerView.bounds

//        blurView1.frame = .zero
//        blurView2.frame = .zero
        let screenW = UIScreen.main.bounds.width
        let scale = 0.65*screenW/402.0
        print("screen width: \(screenW), scale: \(scale)")
        imageContainerView.transform = .init(scaleX: scale, y: scale)
    }
    
    @IBAction func onClickedImageButton(_ sender: UIButton) {
        
        let selectType: ImageType = ImageType(rawValue: sender.tag) ?? .image1
        
        dataManager.currentImage = selectType
        
        imageView.image = UIImage(named: selectType.imageName())
        imageContainerView.currentImage = imageView.image
        
        dataManager.currentImageData = dataManager.loadCurrentImageData()
        dataManager.preloadData()

        for button in imageButtons {
                        
            UIView.animate(withDuration: 0.2, delay: 0.0, options: .curveEaseIn) {
                
                if sender == button {
                    
                    print("clicked button tag: \(button.tag)")

                    button.backgroundColor = .systemBlue
                    button.setTitleColor(.white, for: .normal)
                    
                } else {
                    button.backgroundColor = .white
                    button.setTitleColor(.systemBlue, for: .normal)
                }
                
                self.blurView1.isHidden = (sender == self.imageButton7)
                self.blurView2.isHidden = (sender == self.imageButton7)
                self.blurView3.isHidden = (sender == self.imageButton7)
                self.blurView4.isHidden = (sender == self.imageButton7)

            }
        }
        
//        imageContainerView.redrawDisplay()
    }
}

