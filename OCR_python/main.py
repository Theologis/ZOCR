def main ():
    from utils import RGBtoHSV
    from utils import GetBoundingBoxes
    from extract_letters import FindLetters
    from utils import BoundingBoxes_from_Points

    proj_directory = '/Users/Theologis/Desktop/CAS Lab/OCR/' ##TOCHANGE...
    image_name='sample1.jpg' ##TOCHANGE...

    image=RGBtoHSV(proj_directory,image_name,sensitivity = 24)

    
    threshold_area = 30
    start_point,end_point =  GetBoundingBoxes(image,threshold_area)

    start_point,end_point = FindLetters(start_point,end_point,image)
    BoundingBoxes_from_Points(start_point,end_point,image)
