from OCC.Core.TColgp import TColgp_HArray1OfPnt2d, TColgp_Array1OfPnt2d,TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt2d,gp_Vec, gp_Pnt
from OCC.Core.Geom import Geom_BezierSurface, Geom_BSplineSurface
from OCC.Core.TColGeom import TColGeom_Array2OfBezierSurface
from OCC.Core.GeomConvert import GeomConvert_CompBezierSurfacesToBSplineSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Display.SimpleGui import init_display
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static_SetCVal
import numpy as np

def example_vis(surf):
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.EraseAll()
    display.DisplayShape(surf, update=True)
    start_display()

def cp_to_bz(cp):
    """Tensor Product (Bezier) Control Points to OCC-BezierSurface object.

    Args:
        cp (np.array): input control point, 4x4x3

    Returns:
        OCC-Geom_BezierSurface: OCC internal type for a Bezier Patch
    """
    array1 = TColgp_Array2OfPnt(1, len(cp), 1, len(cp))
    for i in range(1,len(cp)+1):
        for j in range(1,len(cp)+1):
            array1.SetValue(i,j, gp_Pnt(*cp[i-1,j-1]))
    BZ1 = Geom_BezierSurface(array1)
    return BZ1

def cp_write_to_step(output_file, quad_cp):
    step_writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", "AP203")

    for cp in quad_cp:
        assert len(cp) == 16
        b1 = cp_to_bz(cp.reshape(4,4,3))
        build = BRepBuilderAPI_MakeFace(b1, 1e-6)
        step_writer.Transfer(build.Shape(),STEPControl_AsIs)
    status = step_writer.Write(output_file)


def compose_bezier(bz_list):
    bezierarray = TColGeom_Array2OfBezierSurface(1, len(bz_list), 1,1)
    for i,b in enumerate(bz_list):
        bezierarray.SetValue(i+1, 1, b)
    BB = GeomConvert_CompBezierSurfacesToBSplineSurface(bezierarray)
    if BB.IsDone():
        poles = BB.Poles().Array2()
        uknots = BB.UKnots().Array1()
        vknots = BB.VKnots().Array1()
        umult = BB.UMultiplicities().Array1()
        vmult = BB.VMultiplicities().Array1()
        udeg = BB.UDegree()
        vdeg = BB.VDegree()
        BSPLSURF = Geom_BSplineSurface( poles, uknots, vknots, umult, vmult, udeg, vdeg, False, False)

        return BSPLSURF
    else:
        return None

def test_compose():
    cp = np.zeros((4,4,3))
    cp2 = np.zeros((4,4,3))
    for i in range(4):
        for j in range(4):
            cp[i,j] = (i-3, j,(i-3)**2)
            cp2[i,j] = (i,j,i**2)
    b1 = cp_to_bz(cp)
    b2 = cp_to_bz(cp2)
    bsp = compose_bezier([b1,b2])

    step_writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", "AP203")

    build = BRepBuilderAPI_MakeFace(bsp, 1e-6)
    step_writer.Transfer(build.Shape(),STEPControl_AsIs)
    status = step_writer.Write('test.stp')
    
def stripe_writer(out_file, all_stripes, quad_cp):
    def rotate(cp, e):
        return np.rot90(cp.reshape(4,4,3), k=-e)
    step_writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", "AP203")

    for stripe in all_stripes:
        s0cp = np.array([rotate(quad_cp[f],e) for f,e in stripe])
        bzlist = [cp_to_bz(s) for s in s0cp]
        if len(stripe) > 1:
            bb = compose_bezier(bzlist)
        else:
            bb = bzlist[0]
        step_writer.Transfer(BRepBuilderAPI_MakeFace(bb, 1e-6).Shape(),
                            STEPControl_AsIs)
    status = step_writer.Write(out_file)
    return status