import cv2
import numpy as np
import glob
import os

class PanaromaStitcher:
    def __init__(self):
        pass

    def compute_homography(self, points1, points2):
        """Simple homography computation between two sets of points"""
        points1 = points1[:, :2].reshape(-1, 1, 2)
        points2 = points2[:, :2].reshape(-1, 1, 2)
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
        return H

    def stitch_two_images(self, img1, img2):
        """Stitch two consecutive images together"""
        # Find features
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # Match features
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)

        # Filter matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Get matching points
        points1 = np.float32([kp1[m.queryIdx].pt + (1,) for m in good_matches])
        points2 = np.float32([kp2[m.trainIdx].pt + (1,) for m in good_matches])

        # Get homography
        H = self.compute_homography(points1, points2)

        # Calculate panorama dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        corners1_transform = cv2.perspectiveTransform(corners1, H)
        corners = np.concatenate((corners2, corners1_transform), axis=0)
        
        [xmin, ymin] = np.int32(corners.min(axis=0).ravel())
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel())
        
        # Adjust homography
        translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
        H = translation @ H

        # Create panorama
        output = np.zeros((ymax-ymin, xmax-xmin, 3), dtype=np.uint8)
        warped = cv2.warpPerspective(img1, H, (xmax-xmin, ymax-ymin))
        output[-ymin:h2-ymin, -xmin:w2-xmin] = img2

        # Simple blending
        mask = (warped != 0) & (output != 0)
        output[mask] = (warped[mask] + output[mask]) // 2
        mask = (output == 0)
        output[mask] = warped[mask]

        return H, output

    def make_panaroma_for_images_in(self, path):
        """Create panorama from images in directory"""
        # Get and sort images
        image_paths = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(image_paths)} images for stitching')
        
        # Read images
        images = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                print(f'\t\t reading... {img_path} having size {img.shape}')

        if len(images) < 2:
            return images[0] if images else None, []

        # Stitch images sequentially
        homography_matrices = []
        result = images[0]
        
        for i in range(1, len(images)):
            H, result = self.stitch_two_images(result, images[i])
            homography_matrices.append(H)
            print(f'Stitched image pair {i}')

        return result, homography_matrices

if __name__ == '__main__':
    # Test code
    stitcher = PanaromaStitcher()
    test_path = 'test_imgs'  # Replace with your test path
    result, homographies = stitcher.make_panaroma_for_images_in(test_path)
    if result is not None:
        cv2.imwrite('panorama_result.jpg', result)