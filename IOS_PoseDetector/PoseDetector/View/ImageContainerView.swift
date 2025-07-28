//
//  ImageContainerView.swift
//  PoseDetector
//
//  
//

import UIKit

class ImageContainerView: UIView {
    
    var currentImage: UIImage?
    
    func redrawDisplay() {

        if let img = currentImage {
            
            let w = Int(self.bounds.size.width)
            let h = Int(self.bounds.size.height)

            guard let context = UIGraphicsGetCurrentContext() else {
                return
            }
            
            // 
            context.setFillColor(UIColor.red.cgColor)
            context.setStrokeColor(UIColor.black.cgColor)
            context.setLineWidth(2)
            
            for x in 0..<w {
                
                for y in 0..<h {
                    
                    let p = CGPoint(x: x, y: y)
                    if img.isBluePoint(p, frameSize: self.bounds.size) {
                        
                        let circle = CGRect(x: p.x - 1, y: p.y - 1, width: 2, height: 2)
                        context.fillEllipse(in: circle)
                        context.strokeEllipse(in: circle)
                    }
                }
            }
        }
    }
    
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        
        guard let point = touches.first?.location(in: self) else {
            return
        }
        
        handleTouch(point)
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        
        guard let point = touches.first?.location(in: self) else {
            return
        }
        
        handleTouch(point)
    }
    
    //
    func handleTouch(_ point: CGPoint) {
        
        guard let image = self.currentImage else {
            return
        }
        
        if image.isBluePoint(point, frameSize: self.bounds.size) {
            
            let reflectPoint = reflectPoint(point)
            
//            print("🟩 body: \(reflectPoint)")
            
            if let nearbyPoint = DataManager.shared.findNearbyPoint(reflectPoint) {
                
                var feedbackType: UIImpactFeedbackGenerator.FeedbackStyle = .light

                switch nearbyPoint.vibration {
                case 1:
                    feedbackType = .light
                case 2:
                    feedbackType = .medium
                case 3:
                    feedbackType = .heavy
                default:
                    feedbackType = .light
                }
                
//                print("feedbackType: \(feedbackType)")
                
                triggerImpactFeedback(feedbackType)
            } else {
                triggerImpactFeedback(.light)
            }
        }
    }
    
    
    

    let constX = 1.6
    let constY = 1.0
    func reflectPoint(_ p: CGPoint) -> CGPoint {
        
        let w = self.bounds.width
        let h = self.bounds.height
        
        
        let newPoint = CGPoint(x: ((2*p.x/w - 1)*constX).rounded(toPlaces: 6), y: ((2*p.y/h - 1)*constY).rounded(toPlaces: 6))
        
        return newPoint
    }
    
    func triggerImpactFeedback(_ style: UIImpactFeedbackGenerator.FeedbackStyle) {
        let generator = UIImpactFeedbackGenerator(style: style)
        generator.impactOccurred()
    }
}

extension Double {
    func rounded(toPlaces places: Int) -> CGFloat {
        let divisor = pow(10.0, CGFloat(places))
        return (self * divisor).rounded() / divisor
    }
}
