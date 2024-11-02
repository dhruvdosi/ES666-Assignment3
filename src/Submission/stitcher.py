import cv2
import numpy as np
import glob
import os

class PanaromaStitcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        self.sift = cv2.SIFT_create(nfeatures=5000)  # Increased features

    def cylindrical_warp(self, img, f):
        """Apply cylindrical warp to image"""
        h, w = img.shape[:2]
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])  # Camera matrix
        
        # Create 3D points
        y_i, x_i = np.indices((h, w))
        X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h*w, 3)
        
        # Project to cylinder
        Kinv = np.linalg.inv(K)
        X = Kinv.dot(X.T).T
        theta = np.arctan2(X[:, 0], X[:, 2])
        h_i = X[:, 1] / np.sqrt(X[:, 0]**2 + X[:, 2]**2)
        
        # Project back to image
        x_proj = f * theta
        y_proj = f * h_i
        
        # Convert to pixel coordinates
        x_proj = x_proj + w/2
        y_proj = y_proj + h/2
        
        x_proj = x_proj.reshape(h, w)
        y_proj = y_proj.reshape(h, w)
        
        # Remap
        warped = cv2.remap(img, 
                          x_proj.astype(np.float32),
                          y_proj.astype(np.float32),
                          cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
        return warped

    def compute_homography(self, src_pts, dst_pts):
        """Compute homography with RANSAC and additional checks"""
        if len(src_pts) < 4:
            return None
            
        # Normalize points
        src_mean = np.mean(src_pts[:, :2], axis=0)
        dst_mean = np.mean(dst_pts[:, :2], axis=0)
        src_std = np.std(src_pts[:, :2], axis=0)
        dst_std = np.std(dst_pts[:, :2], axis=0)
        
        T_src = np.array([[1/src_std[0], 0, -src_mean[0]/src_std[0]],
                         [0, 1/src_std[1], -src_mean[1]/src_std[1]],
                         [0, 0, 1]])
        T_dst = np.array([[1/dst_std[0], 0, -dst_mean[0]/dst_std[0]],
                         [0, 1/dst_std[1], -dst_mean[1]/dst_std[1]],
                         [0, 0, 1]])
        
        # Transform points
        src_norm = np.dot(T_src, np.hstack((src_pts[:, :2], np.ones((len(src_pts), 1)))).T).T
        dst_norm = np.dot(T_dst, np.hstack((dst_pts[:, :2], np.ones((len(dst_pts), 1)))).T).T
        
        # Compute homography using normalized points
        H, mask = cv2.findHomography(src_norm[:, :2], dst_norm[:, :2], 
                                   cv2.RANSAC, 5.0)
        
        if H is None:
            return None
            
        # Denormalize
        H = np.dot(np.linalg.inv(T_dst), np.dot(H, T_src))
        
        # Check if homography is valid
        if (np.abs(H[2, 2]) < 1e-6 or 
            np.abs(H[2, 0]) > 0.1 or 
            np.abs(H[2, 1]) > 0.1):
            return None
        
        return H

    def match_features(self, des1, des2):
        """Improved feature matching with ratio test and cross-checking"""
        if des1 is None or des2 is None:
            return []
            
        # Compute matches both ways
        matches1 = self.matcher.knnMatch(des1, des2, k=2)
        matches2 = self.matcher.knnMatch(des2, des1, k=2)
        
        # Apply ratio test both ways
        good_matches1 = []
        for m, n in matches1:
            if m.distance < 0.7 * n.distance:
                good_matches1.append(m)
                
        good_matches2 = []
        for m, n in matches2:
            if m.distance < 0.7 * n.distance:
                good_matches2.append(m)
        
        # Cross-check
        good_matches = []
        for match1 in good_matches1:
            for match2 in good_matches2:
                if match1.queryIdx == match2.trainIdx and match1.trainIdx == match2.queryIdx:
                    good_matches.append(match1)
                    break
                    
        return good_matches

    def stitch_two_images(self, img1, img2):
        """Stitch two images with improved matching and blending"""
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
        
        # Add homogeneous coordinate
        src_pts = np.hstack((src_pts, np.ones((len(src_pts), 1))))
        dst_pts = np.hstack((dst_pts, np.ones((len(dst_pts), 1))))
        
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
        
        # Adjust homography
        translation = np.array([[1, 0, -xmin], 
                              [0, 1, -ymin],
                              [0, 0, 1]])
        H = translation @ H
        
        # Create panorama with alpha blending
        warped = cv2.warpPerspective(img1, H, (xmax-xmin, ymax-ymin))
        output = np.zeros((ymax-ymin, xmax-xmin, 3), dtype=np.uint8)
        output[-ymin:h2-ymin, -xmin:w2-xmin] = img2
        
        # Create masks for blending
        mask1 = cv2.warpPerspective(np.ones_like(gray1), H, (xmax-xmin, ymax-ymin))
        mask2 = np.zeros((ymax-ymin, xmax-xmin), dtype=np.float32)
        mask2[-ymin:h2-ymin, -xmin:w2-xmin] = 1
        
        # Blend images
        mask_intersection = (mask1 > 0) & (mask2 > 0)
        if np.any(mask_intersection):
            # Create gradual blending weights
            fade_width = 50  # Width of blending region
            weight1 = np.zeros_like(mask1, dtype=np.float32)
            weight2 = np.zeros_like(mask2, dtype=np.float32)
            
            for i in range(fade_width):
                weight1 += (mask1 > 0) & (mask2 == 0)
                mask1 = cv2.erode(mask1, np.ones((3,3), np.uint8))
                weight2 += (mask2 > 0) & (mask1 == 0)
                mask2 = cv2.erode(mask2, np.ones((3,3), np.uint8))
                
            weight1 = cv2.GaussianBlur(weight1, (0,0), fade_width/3)
            weight2 = cv2.GaussianBlur(weight2, (0,0), fade_width/3)
            
            total_weight = weight1 + weight2
            weight1 /= total_weight
            weight2 /= total_weight
            
            for c in range(3):
                output[...,c] = (warped[...,c] * weight1 + 
                                output[...,c] * weight2)
        
        return H, output

    def make_panaroma_for_images_in(self, path, f=None):
        """Create panorama from images with automatic focal length estimation"""
        # Get and sort images
        image_paths = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(image_paths)} images for stitching')
        
        if len(image_paths) < 2:
            return None, []
            
        # Read and preprocess images
        images = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Estimate focal length if not provided
            if f is None:
                # Use image diagonal as approximate focal length
                h, w = img.shape[:2]
                f = np.sqrt(h*h + w*w)
            
            # Apply cylindrical warp
            img = self.cylindrical_warp(img, f)
            images.append(img)
            print(f'\t\t reading... {img_path} having size {img.shape}')
            
        if len(images) < 2:
            return images[0] if images else None, []
            
        # Start from middle image
        mid = len(images) // 2
        result = images[mid]
        homography_matrices = []
        
        # Stitch images to the right
        for i in range(mid + 1, len(images)):
            H, stitched = self.stitch_two_images(result, images[i])
            if stitched is not None:
                result = stitched
                homography_matrices.append(H)
                print(f'Stitched right image {i}')
                
        # Stitch images to the left
        temp_result = images[mid]
        for i in range(mid - 1, -1, -1):
            H, stitched = self.stitch_two_images(images[i], temp_result)
            if stitched is not None:
                temp_result = stitched
                homography_matrices.append(H)
                print(f'Stitched left image {i}')
        
        # If we have both left and right stitched images, combine them
        if temp_result is not None and temp_result is not images[mid]:
            H, result = self.stitch_two_images(temp_result, result)
            if H is not None:
                homography_matrices.append(H)
                
        return result, homography_matrices

if __name__ == '__main__':
    stitcher = PanaromaStitcher()
    test_path = 'test_imgs'
    result, homographies = stitcher.make_panaroma_for_images_in(test_path)
    if result is not None:
        cv2.imwrite('panorama_result.jpg', result)