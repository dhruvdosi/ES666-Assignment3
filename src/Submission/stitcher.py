import cv2
import numpy as np
import glob
import os

class PanaromaStitcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        self.sift = cv2.SIFT_create(nfeatures=2000)

    def cylindrical_warp(self, img, f):
        """Apply cylindrical warp to image"""
        h, w = img.shape[:2]
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
        
        y_i, x_i = np.indices((h, w))
        X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h*w, 3)
        
        Kinv = np.linalg.inv(K)
        X = Kinv.dot(X.T).T
        theta = np.arctan2(X[:, 0], X[:, 2])
        h_i = X[:, 1] / np.sqrt(X[:, 0]**2 + X[:, 2]**2)
        
        x_proj = f * theta
        y_proj = f * h_i
        
        x_proj = x_proj + w/2
        y_proj = y_proj + h/2
        
        x_proj = x_proj.reshape(h, w)
        y_proj = y_proj.reshape(h, w)
        
        warped = cv2.remap(img, 
                          x_proj.astype(np.float32),
                          y_proj.astype(np.float32),
                          cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
        return warped

    def compute_homography(self, src_pts, dst_pts):
        """Compute homography with improved RANSAC and validation"""
        if len(src_pts) < 4:
            return None
            
        # Compute homography using RANSAC
        H, mask = cv2.findHomography(src_pts[:, :2], dst_pts[:, :2], 
                                   cv2.RANSAC, 
                                   ransacReprojThreshold=5.0,
                                   maxIters=2000,
                                   confidence=0.995)
        
        if H is None:
            return None
            
        # Validate homography
        if (np.abs(H[2, 2]) < 1e-6 or 
            np.abs(H[2, 0]) > 0.1 or 
            np.abs(H[2, 1]) > 0.1 or
            np.abs(np.linalg.det(H)) < 1e-6):
            return None
        
        return H

    def match_features(self, des1, des2):
        """Improved feature matching with bidirectional ratio test"""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        
        matches1 = self.matcher.knnMatch(des1, des2, k=2)
        matches2 = self.matcher.knnMatch(des2, des1, k=2)
        
        # Filter matches using ratio test
        good_matches1 = []
        for m, n in matches1:
            if m.distance < 0.75 * n.distance:
                good_matches1.append(m)
                
        good_matches2 = []
        for m, n in matches2:
            if m.distance < 0.75 * n.distance:
                good_matches2.append(m)
        
        # Cross-check matches
        good_matches = []
        for match1 in good_matches1:
            for match2 in good_matches2:
                if match1.queryIdx == match2.trainIdx and match1.trainIdx == match2.queryIdx:
                    good_matches.append(match1)
                    break
                    
        return good_matches

    def blend_images(self, warped, output, mask1, mask2):
        """Improved image blending with numerical stability checks"""
        h, w = output.shape[:2]
        result = np.copy(output)
        
        # Create blending masks
        overlap = (mask1 > 0) & (mask2 > 0)
        if not np.any(overlap):
            result = np.where(mask1[..., None] > 0, warped, output)
            return result
            
        # Create distance transforms for smooth blending
        dist1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 3)
        dist2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 3)
        
        # Normalize weights
        weight1 = dist1 / (dist1 + dist2 + 1e-6)
        weight2 = dist2 / (dist1 + dist2 + 1e-6)
        
        # Ensure weights sum to 1
        total = weight1 + weight2
        weight1 = np.divide(weight1, total, out=np.zeros_like(weight1), where=total > 0)
        weight2 = np.divide(weight2, total, out=np.zeros_like(weight2), where=total > 0)
        
        # Apply blending
        for c in range(3):
            result[..., c] = (warped[..., c] * weight1 + output[..., c] * weight2)
        
        return result.astype(np.uint8)

    def stitch_two_images(self, img1, img2):
        """Stitch two images with improved error handling"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
        if len(kp1) < 10 or len(kp2) < 10:
            return None, None
        
        # Match features
        matches = self.match_features(des1, des2)
        
        if len(matches) < 10:
            return None, None
            
        # Get matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Compute homography
        H = self.compute_homography(src_pts, dst_pts)
        
        if H is None:
            return None, None
            
        # Calculate panorama dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        corners1_transform = cv2.perspectiveTransform(corners1, H)
        corners = np.concatenate((corners2, corners1_transform), axis=0)
        
        [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
        
        # Ensure positive dimensions
        if xmax <= xmin or ymax <= ymin:
            return None, None
        
        # Adjust homography
        translation = np.array([[1, 0, -xmin], 
                              [0, 1, -ymin],
                              [0, 0, 1]])
        H = translation @ H
        
        # Warp and create output image
        warped = cv2.warpPerspective(img1, H, (xmax-xmin, ymax-ymin))
        output = np.zeros((ymax-ymin, xmax-xmin, 3), dtype=np.uint8)
        output[-ymin:h2-ymin, -xmin:w2-xmin] = img2
        
        # Create masks
        mask1 = cv2.warpPerspective(np.ones_like(gray1), H, (xmax-xmin, ymax-ymin))
        mask2 = np.zeros((ymax-ymin, xmax-xmin), dtype=np.float32)
        mask2[-ymin:h2-ymin, -xmin:w2-xmin] = 1
        
        # Blend images
        result = self.blend_images(warped, output, mask1, mask2)
        
        return H, result

    def make_panaroma_for_images_in(self, path, f=None):
        """Create panorama from images with improved error handling"""
        image_paths = sorted(glob.glob(os.path.join(path, '*')))
        print(f'Found {len(image_paths)} images for stitching')
        
        if len(image_paths) < 2:
            return None, []
            
        # Read and preprocess images
        images = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            if f is None:
                h, w = img.shape[:2]
                f = np.sqrt(h*h + w*w)
            
            img = self.cylindrical_warp(img, f)
            images.append(img)
            print(f'\t\t reading... {img_path} having size {img.shape}')
            
        if len(images) < 2:
            return images[0] if images else None, []
            
        # Start from middle image
        mid = len(images) // 2
        result = images[mid]
        homography_matrices = []
        
        # Stitch right images
        for i in range(mid + 1, len(images)):
            H, stitched = self.stitch_two_images(result, images[i])
            if stitched is not None:
                result = stitched
                homography_matrices.append(H)
                print(f'Stitched right image {i}')
                
        # Stitch left images
        temp_result = images[mid]
        for i in range(mid - 1, -1, -1):
            H, stitched = self.stitch_two_images(images[i], temp_result)
            if stitched is not None:
                temp_result = stitched
                homography_matrices.append(H)
                print(f'Stitched left image {i}')
        
        # Combine left and right panoramas
        if temp_result is not None and not np.array_equal(temp_result, images[mid]):
            H, final_result = self.stitch_two_images(temp_result, result)
            if H is not None and final_result is not None:
                result = final_result
                homography_matrices.append(H)
        
        return result, homography_matrices

if __name__ == '__main__':
    stitcher = PanaromaStitcher()
    test_path = 'test_imgs'
    result, homographies = stitcher.make_panaroma_for_images_in(test_path)
    
    if result is not None:
        # Ensure output directory exists
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'panorama_result.jpg')
        cv2.imwrite(output_path, result)
        print(f'Panorama saved successfully to {output_path}')
    else:
        print('Failed to create panorama')
