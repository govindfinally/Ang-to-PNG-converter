import numpy as np
from PIL import Image
import ezdxf
import cv2
from skimage import measure, morphology
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt

class CleanMicrostructureDXF:
    def __init__(self, image_path, output_path="clean_microstructure.dxf", scale_factor=1.0):
        """
        Create clean, COMSOL-ready DXF from segmented microstructure
        
        Args:
            image_path: Path to segmented microstructure image
            output_path: Output DXF path  
            scale_factor: Real-world scale (units per pixel)
        """
        self.image_path = image_path
        self.output_path = output_path
        self.scale_factor = scale_factor
        self.doc = ezdxf.new('R2010')
        self.msp = self.doc.modelspace()
        
    def create_grain_label_map(self):
        """Convert color image to integer grain labels"""
        print("Creating grain label map...")
        
        # Load image
        image = Image.open(self.image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
        
        # Get unique colors and create label map
        h, w = img_array.shape[:2]
        pixels = img_array.reshape(-1, 3)
        
        # Find unique colors
        unique_colors = []
        color_to_label = {}
        label = 0
        
        for pixel in pixels:
            pixel_tuple = tuple(pixel)
            if pixel_tuple not in color_to_label:
                color_to_label[pixel_tuple] = label
                unique_colors.append(pixel_tuple)
                label += 1
        
        # Create label map
        label_map = np.zeros((h, w), dtype=np.int32)
        for i in range(h):
            for j in range(w):
                pixel_tuple = tuple(img_array[i, j])
                label_map[i, j] = color_to_label[pixel_tuple]
        
        print(f"Found {len(unique_colors)} unique grains")
        return label_map, unique_colors
    
    def extract_clean_grain_boundaries(self, min_area=100):
        """Extract one clean boundary per grain"""
        print("Extracting grain boundaries...")
        
        # Get grain labels
        label_map, unique_colors = self.create_grain_label_map()
        h, w = label_map.shape
        
        grain_boundaries = []
        
        for grain_id in range(len(unique_colors)):
            # Create binary mask for this grain
            grain_mask = (label_map == grain_id).astype(np.uint8)
            
            # Skip small grains
            if np.sum(grain_mask) < min_area:
                continue
            
            # Clean up mask
            kernel = np.ones((3, 3), np.uint8)
            grain_mask = cv2.morphologyEx(grain_mask, cv2.MORPH_CLOSE, kernel)
            grain_mask = cv2.medianBlur(grain_mask, 3)
            
            # Find the largest contour only
            contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) >= min_area:
                    # Simplify contour significantly
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    if len(simplified_contour) >= 3:
                        # Convert to scaled coordinates (flip Y)
                        points = []
                        for pt in simplified_contour:
                            x = float(pt[0][0]) * self.scale_factor
                            y = float((h - pt[0][1])) * self.scale_factor
                            points.append((x, y))
                        
                        grain_boundaries.append({
                            'id': grain_id,
                            'color': unique_colors[grain_id],
                            'points': points,
                            'area': cv2.contourArea(largest_contour)
                        })
        
        print(f"Extracted {len(grain_boundaries)} clean grain boundaries")
        return grain_boundaries
    
    def create_simple_dxf(self, min_area=100):
        """Create simple DXF with only essential boundaries"""
        print("Creating simple DXF for COMSOL...")
        
        # Get clean boundaries
        grain_boundaries = self.extract_clean_grain_boundaries(min_area)
        
        if not grain_boundaries:
            print("❌ No grain boundaries found!")
            return None
        
        # Create only boundary lines (no fills, no hatches)
        grain_count = 0
        for grain in grain_boundaries:
            layer_name = f"GRAIN_{grain_count:03d}"
            
            # Create layer with unique color
            color_id = (grain_count % 254) + 1
            self.doc.layers.new(name=layer_name, dxfattribs={'color': color_id})
            
            # Add only the boundary line
            if len(grain['points']) >= 3:
                try:
                    # Create simple polyline boundary
                    polyline = self.msp.add_lwpolyline(grain['points'], close=True)
                    polyline.dxf.layer = layer_name
                    polyline.dxf.flags = 1  # Closed
                    
                    grain_count += 1
                    
                except Exception as e:
                    print(f"Warning: Skipped grain {grain['id']}: {e}")
                    continue
        
        # Add simple outer boundary
        self._add_simple_boundary()
        
        # Save DXF
        try:
            self.doc.saveas(self.output_path)
            print(f"✅ Clean DXF saved: {self.output_path}")
            
            return {
                'total_grains': grain_count,
                'boundaries': grain_boundaries
            }
            
        except Exception as e:
            print(f"❌ Error saving DXF: {e}")
            return None
    
    def _add_simple_boundary(self):
        """Add simple outer rectangle"""
        image = Image.open(self.image_path)
        width, height = image.size
        
        # Outer boundary points
        boundary_points = [
            (0, 0),
            (width * self.scale_factor, 0),
            (width * self.scale_factor, height * self.scale_factor),
            (0, height * self.scale_factor)
        ]
        
        # Create boundary layer
        self.doc.layers.new(name='BOUNDARY', dxfattribs={'color': 7})  # White
        
        # Add boundary rectangle
        boundary = self.msp.add_lwpolyline(boundary_points, close=True)
        boundary.dxf.layer = 'BOUNDARY'
        boundary.dxf.flags = 1

def create_comsol_instructions():
    """Generate specific COMSOL import instructions"""
    return """
=== COMSOL IMPORT INSTRUCTIONS ===

STEP 1: Import DXF
------------------
1. File → Import → For Geometry
2. Select your DXF file
3. Import Settings:
   - Geometry import: Import
   - Length unit: Set to your scale (e.g., µm)
   - Repair tolerance: 1e-6

STEP 2: Create Work Plane  
------------------------
1. Right-click Geometry → Work Plane
2. Plane type: Face parallel
3. Select xy-plane

STEP 3: Convert to Regions
-------------------------
1. Right-click Work Plane → Conversions → Convert to Plane
2. Select all imported curves
3. Right-click Work Plane → Conversions → Convert to Solid
4. Build Geometry (F8)

STEP 4: Check Results
--------------------
- You should see solid regions, not lines
- Each grain should be a separate domain
- No overlapping lines or mesh artifacts

TROUBLESHOOTING:
- If still seeing lines: Use "Create Regions" instead of Convert to Solid
- If regions don't form: Check that all boundaries are closed
- If import fails: Try saving DXF as R2010 format
"""

def plot_clean_boundaries(image_path, grain_boundaries, output_path):
    """Plot the clean boundaries for verification"""
    print("Creating boundary verification plot...")
    
    # Load original image
    image = Image.open(image_path)
    img_array = np.array(image)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    ax1.imshow(img_array)
    ax1.set_title('Original Microstructure')
    ax1.axis('off')
    
    # Clean boundaries
    ax2.imshow(img_array, alpha=0.6)
    
    # Draw only the clean boundaries
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, grain in enumerate(grain_boundaries):
        points = np.array(grain['points'])
        # Flip Y back for display
        points[:, 1] = img_array.shape[0] - points[:, 1]
        
        # Close polygon
        points = np.vstack([points, points[0]])
        
        color = colors[i % len(colors)]
        ax2.plot(points[:, 0], points[:, 1], color=color, linewidth=3, alpha=0.8)
        
        # Add grain number
        centroid_x = np.mean(points[:-1, 0])
        centroid_y = np.mean(points[:-1, 1])
        ax2.text(centroid_x, centroid_y, str(i), 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='white', alpha=0.9))
    
    ax2.set_title(f'Clean Boundaries ({len(grain_boundaries)} grains)')
    ax2.axis('off')
    
    plt.tight_layout()
    plot_path = output_path.replace('.dxf', '_clean_boundaries.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Boundary plot saved: {plot_path}")

def main():
    """Main execution with error handling"""
    
    # ===== CONFIGURATION =====
    input_image =  r"C:\Users\HP\OneDrive\Desktop\Slice1.png.png"  # UPDATE PATH
    output_dxf = r"C:\Users\HP\OneDrive\Desktop\hai_clean_microstructure.dxf"  # UPDATE PATH
    
    # Parameters
    scale_factor = 1.0  # Adjust as needed (units per pixel)
    min_grain_area = 200  # Minimum grain area in pixels
    
    print("=== CLEAN MICROSTRUCTURE TO DXF ===\n")
    
    try:
        # Create converter
        converter = CleanMicrostructureDXF(input_image, output_dxf, scale_factor)
        
        # Create simple DXF
        result = converter.create_simple_dxf(min_area=min_grain_area)
        
        if result:
            print(f"\n✅ SUCCESS!")
            print(f"✅ Created {result['total_grains']} grain regions")
            print(f"✅ DXF saved: {output_dxf}")
            
            # Create verification plot
            plot_clean_boundaries(input_image, result['boundaries'], output_dxf)
            
            # Print instructions
            print(create_comsol_instructions())
            
        else:
            print("❌ Failed to create DXF")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Required: numpy, pillow, ezdxf, opencv-python, scikit-image, matplotlib")
    print("Install: pip install numpy pillow ezdxf opencv-python scikit-image matplotlib\n")
    
    main()