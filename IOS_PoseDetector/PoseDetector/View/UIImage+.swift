//
//  UIImage+.swift
//  PoseDetector
//

//

import UIKit

extension UIImage {
    
    func getPixelColor(in pos: CGPoint, frameSize size: CGSize) -> UIColor {
        let x: CGFloat = (self.size.width) * pos.x / size.width
        let y: CGFloat = (self.size.height) * pos.y / size.height
        let pixelPoint: CGPoint = CGPoint(x: x, y: y)
        let pixelData = self.cgImage!.dataProvider!.data
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        let pixelInfo: Int = ((Int(self.size.width) * Int(pixelPoint.y)) + Int(pixelPoint.x)) * 4
        
        let r = CGFloat(data[pixelInfo]) / CGFloat(255.0)
        let g = CGFloat(data[pixelInfo+1]) / CGFloat(255.0)
        let b = CGFloat(data[pixelInfo+2]) / CGFloat(255.0)
        let a = CGFloat(data[pixelInfo+3]) / CGFloat(255.0)
        
        let color = UIColor(red: r, green: g, blue: b, alpha: a)
        return color
    }
    
    func isBluePoint(_ point: CGPoint, frameSize size: CGSize) -> Bool {
        let x: CGFloat = (self.size.width) * point.x / size.width
        let y: CGFloat = (self.size.height) * point.y / size.height
        let pixelPoint: CGPoint = CGPoint(x: x, y: y)
        let pixelData = self.cgImage!.dataProvider!.data
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        let pixelInfo: Int = ((Int(self.size.width) * Int(pixelPoint.y)) + Int(pixelPoint.x)) * 4
        
        let r = CGFloat(data[pixelInfo]) / CGFloat(255.0)
        let g = CGFloat(data[pixelInfo+1]) / CGFloat(255.0)
        let b = CGFloat(data[pixelInfo+2]) / CGFloat(255.0)
        let a = CGFloat(data[pixelInfo+3]) / CGFloat(255.0)
        
        let isBlue = (r < 0.5) && (g < 0.5) && (b > 0.5) && a > 0.5

        //        print("p: \(point) - {\(r), \(g), \(b)} - blue: \(isBlue)")

        return isBlue
    }
}
