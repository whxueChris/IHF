//
//  DataManager.swift
//  PoseDetector
//
//  
//

import Foundation

enum ImageType: Int {
    case image1
    case image2
    case image3
    case image4
    case image5
    case image6
    case image7

    func imageName() -> String {
        switch self {
        case .image1:
            return "output_image1"
        case .image2:
            return "output_image2"
        case .image3:
            return "output_image3"
        case .image4:
            return "output_image4"
        case .image5:
            return "output_image5"
        case .image6:
            return "output_image6"
        case .image7:
            return "output_image7"
        }
    }
    
    func dataName() -> String {
        switch self {
        case .image1:
            return "k_means_data_final1.json"
        case .image2:
            return "k_means_data_final2.json"
        case .image3:
            return "k_means_data_final3.json"
        case .image4:
            return "k_means_data_final4.json"
        case .image5:
            return "k_means_data_final5.json"
        case .image6:
            return "k_means_data_final6.json"
        case .image7:
            return "k_means_data_final7.json"
        }
    }
}

class DataManager {
    
    static let shared = DataManager()
    
    var currentImage: ImageType = .image1
    var currentImageData: [ImagePoint] = []
    var currentImagePointKeys: [Double] = []
    var currentImageDataMapper: [CGFloat: [ImagePoint]] = [:]

    
    func loadCurrentImageData() -> [ImagePoint] {
        
        let fileName = currentImage.dataName()
                
        guard let url = Bundle.main.url(forResource: fileName, withExtension: "") else {
            return []
        }
        
        do {
            
            let data = try Data(contentsOf: url)
            
           
            let decoder = JSONDecoder()
            
            
            let imagePoints = try decoder.decode([ImagePoint].self, from: data)
            
            return imagePoints
            
        } catch {
           
            print("Failed to decode JSON: \(error)")
            return []
        }
    }
    
    func preloadData() {
        
        
        for point in currentImageData {
            if let _ = currentImageDataMapper[point.x] {
                
                currentImageDataMapper[point.x]?.append(point)
            } else {
                
                currentImagePointKeys.append(point.x)
                
                currentImageDataMapper[point.x] = [point]
            }
        }
        
        currentImagePointKeys.sort()
        
        print(currentImagePointKeys)
    }
    
    func findNearbyPoint(_ reflectPoint: CGPoint) -> ImagePoint? {
        
        let matchX = currentImagePointKeys.reduce(currentImagePointKeys[0]) { abs($0-reflectPoint.x) < abs($1-reflectPoint.x) ? $0 : $1 }
        
        guard let matchYValues = currentImageDataMapper[matchX] else {
            return nil
        }
        
        let matchY = matchYValues.reduce(matchYValues[0]) { abs($0.y-reflectPoint.y) < abs($1.y-reflectPoint.y) ? $0 : $1 }
        
        return matchY
    }
}
