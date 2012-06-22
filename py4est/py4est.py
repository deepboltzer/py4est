#!/usr/bin/env python
# encoding: utf-8

# This file is part of py4est.
# py4est is a python wrapper module for p4est.
# p4est is a C library to manage a collection (a forest) of multiple
# connected adaptive quadtrees or octrees in parallel; see www.p4est.org/.
# 
# py4est copyright (C) 2012 Carsten Burstedde and Aron Ahmadia
# 
# py4est is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# py4est is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with py4est; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

"""
py4est wraps the p4est functionality for static and dynamic AMR in Python.

Currently some important C structs are wrapped with ctypes.
The code relies on the following assumptions about p4est_base.h.
typedef int32_t     p4est_qcoord_t;
typedef int32_t     p4est_topidx_t;
typedef int32_t     p4est_locidx_t;
typedef int64_t     p4est_gloidx_t;
Their python counterparts are defined at the beginning of py4est.py
and they need to be changed whenever the p4est typedefs change.

The p4est core algorithms are documented in
Carsten Burstedde, Lucas C. Wilcox, and Omar Ghattas:
"p4est: Scalable Algorithms for Parallel Adaptive Mesh Refinement on Forests of
Octrees,"
SIAM Journal on Scientific Computing 33 No. 3 (2011), pages 1103-1133.

The AMR pipeline and the interaction between the mesh and the fields that is
behind the conventions in the py4est module are documented in
Carsten Burstedde, Omar Ghattas, Georg Stadler, Tiankai Tu, and Lucas C.
Wilcox:
"Towards Adaptive Mesh PDE Simulations on Petascale Computers,"
Proceedings of Teragrid '08.

If this library turns out useful, we would be grateful for these citations.
"""

import ctypes
import ctypes.util
from mpi4py import MPI

import os

P4EST_DIR=os.getenv('P4EST_DIR')
P4EST_HACK_SOEXT=os.getenv('P4EST_HACK_SOEXT')

#print P4EST_DIR

LIBSCPATH=ctypes.util.find_library(os.path.join(P4EST_DIR,'lib','libsc'))
LIBP4ESTPATH=ctypes.util.find_library(os.path.join(P4EST_DIR,'lib','libp4est'))

#print LIBSCPATH
#print LIBP4ESTPATH

if (P4EST_HACK_SOEXT):
    LIBSCPATH = os.path.join (P4EST_DIR, 'lib', 'libsc.so')
    LIBP4ESTPATH = os.path.join (P4EST_DIR, 'lib', 'libp4est.so')

# Wrap p4est integer typedefs
py4est_int = ctypes.c_int
py4est_int8 = ctypes.c_int8
py4est_size_t = ctypes.c_size_t
py4est_ssize_t = ctypes.c_size_t        # c_ssize_t is only in 2.7
py4est_qcoord = ctypes.c_int32
py4est_topidx = ctypes.c_int32
py4est_locidx = ctypes.c_int32
py4est_gloidx = ctypes.c_int64
py4est_double = ctypes.c_double
py4est_pointer = ctypes.c_void_p

# Wrap p4est composite structures with ctypes.  All indices are 0-based.
class sc_array (ctypes.Structure):
        _fields_ = [("elem_size", py4est_size_t),
                    ("elem_count", py4est_size_t),
                    ("byte_alloc", py4est_ssize_t),
                    ("array", py4est_pointer)]
sc_array_pointer = ctypes.POINTER (sc_array)
class connectivity (ctypes.Structure):
        _fields_ = [("num_vertices", py4est_topidx),
                    ("num_trees", py4est_topidx),
                    ("num_corners", py4est_topidx),
                    ("vertices", ctypes.POINTER (py4est_double)),
                    ("tree_to_vertex", ctypes.POINTER (py4est_topidx)),
                    ("tree_to_attr", ctypes.POINTER (py4est_int8)),
                    ("tree_to_tree", ctypes.POINTER (py4est_topidx)),
                    ("tree_to_face", ctypes.POINTER (py4est_int8)),
                    ("tree_to_corner", ctypes.POINTER (py4est_topidx)),
                    ("ctt_offset", ctypes.POINTER (py4est_topidx)),
                    ("corner_to_tree", ctypes.POINTER (py4est_topidx)),
                    ("corner_to_corner", ctypes.POINTER (py4est_int8))]
connectivity_pointer = ctypes.POINTER (connectivity)
class mesh (ctypes.Structure):
        _fields_ = [("local_num_vertices", py4est_locidx),
                    ("local_num_quadrants", py4est_locidx),
                    ("ghost_num_quadrants", py4est_locidx),
                    ("vertices", py4est_pointer),
                    ("quad_to_vertex", ctypes.POINTER (py4est_locidx)),
                    ("ghost_to_proc", ctypes.POINTER (py4est_int)),
                    ("ghost_to_index", ctypes.POINTER (py4est_locidx)),
                    ("quad_to_quad", ctypes.POINTER (py4est_locidx)),
                    ("quad_to_face", ctypes.POINTER (py4est_int8)),
                    ("quad_to_half", sc_array_pointer)]
mesh_pointer = ctypes.POINTER (mesh)
class wrap (ctypes.Structure):
        _fields_ = [("P4EST_DIM", py4est_int),  # space dimension
                    ("P4EST_HALF", py4est_int), # small faces      2^(dim - 1)
                    ("P4EST_FACES", py4est_int),        # faces    2 * dim
                    ("P4EST_CHILDREN", py4est_int),     # children 2^dim
                    ("conn", connectivity_pointer),
                    ("p4est", py4est_pointer),
                    ("flags", ctypes.POINTER (py4est_int8)),    # one per leaf
                    ("ghost", py4est_pointer),
                    ("mesh", mesh_pointer),
                    ("ghost_aux", py4est_pointer),
                    ("mesh_aux", mesh_pointer),
                    ("match_aux", py4est_int)]  # bool: p4est matches _aux
wrap_pointer = ctypes.POINTER (wrap)

def wrap_get_num_leaves (wrap):
        """This is just for convenience"""
        return wrap.contents.mesh.contents.local_num_quadrants

# Wrap leaf iterator with ctypes
class leaf (ctypes.Structure):
        _fields_ = [("wrap", wrap_pointer),
                    ("level", py4est_int),      # refinement level of leaf
                    ("which_tree", py4est_topidx),      # index of octree
                    ("which_quad", py4est_locidx),      # leaf in this octree
                    ("total_quad", py4est_locidx),      # leaf on this proc
                    ("tree", py4est_pointer),
                    ("quad", py4est_pointer),
                    ("lowerleft", py4est_double * 3),
                    ("upperright", py4est_double * 3)]
leaf_pointer = ctypes.POINTER (leaf)

libsc = ctypes.CDLL(LIBSCPATH, mode=ctypes.RTLD_GLOBAL)
 
# Dynamically link in the  p4est interface
libp4est = ctypes.CDLL (LIBP4ESTPATH)
libp4est.p4est_wrap_new.argtype = py4est_int;
libp4est.p4est_wrap_new.restype = wrap_pointer;
libp4est.p4est_wrap_destroy.argtype = wrap_pointer;
libp4est.p4est_wrap_refine.argtype = wrap_pointer;
libp4est.p4est_wrap_partition.argtype = wrap_pointer;
libp4est.p4est_wrap_complete.argtype = wrap_pointer;
libp4est.p4est_wrap_leaf_first.argtype = wrap_pointer;
libp4est.p4est_wrap_leaf_first.restype = leaf_pointer;
libp4est.p4est_wrap_leaf_next.argtype = leaf_pointer;
libp4est.p4est_wrap_leaf_next.restype = leaf_pointer;

class Py4estDemo:
    def __init__ (self):

        """
        Main idea: Below is a loop that walks through all tree leaves
        on the local processor.  For each leaf, the ctypes-wrapped p4est
        data structures can be queried to obtain neighbor information.
        Based on this information, python objects that represent the
        leaves can be created, and communication patterns can be set up
        to transfer ghost/neighbor values.

        For the encoding convention see the comment blocks in
        p4est_connectivity.h and p4est_mesh.h.  These can be used to
        create lookup tables and permutations between face neighbors.

        Currently, p4est_mesh only records face neighbors, not edges
        and corners.  This will be extended in the near future.
        """

        # Call this once per program before any other p4est function
        libp4est.p4est_wrap_init ()

        # Create a 2D p4est internal state on a square
        initial_level = 1
        self.wrap = libp4est.p4est_wrap_new (initial_level)
        wrapc = self.wrap.contents
        num_leaves = wrap_get_num_leaves (self.wrap)

        # Number of faces of a leaf (4 in 2D, 6 in 3D)
        P4EST_FACES = wrapc.P4EST_FACES
        print "Py faces", P4EST_FACES, "leaves", num_leaves

        # Mesh is the lookup table for leaf neighbors
        # Note that match_aux changes in p4est_wrap_refine and _partition
        # Note that wrap.contents also changes so it is better not to use wrapc
        mesh = wrapc.mesh_aux if wrapc.match_aux else wrapc.mesh

        # Use the leaf iterator to loop over all leafs
        # If only a loop over leaf indices is needed,
        # do instead: for leafindex in range (0, num_leaves)
        leaf = libp4est.p4est_wrap_leaf_first (self.wrap)
        while (leaf):
            # Within this loop the leaf counter is leaf.contents.total_quad
            # All indices in the p4est structures are 0-based

            print "Py leaf level", leaf.contents.level, \
                "tree", leaf.contents.which_tree, \
                "tree_leaf", leaf.contents.which_quad, \
                "local_leaf", leaf.contents.total_quad
            print "Py leaf lower left", \
                leaf.contents.lowerleft[0], leaf.contents.lowerleft[1], \
                "upper right", \
                leaf.contents.upperright[0], leaf.contents.upperright[1]
            for face in range (P4EST_FACES):
                print "Py leaf face", face, "neighbor leaf", \
 mesh.contents.quad_to_quad [P4EST_FACES * leaf.contents.total_quad + face]

            # Advance the leaf iteration
            leaf = libp4est.p4est_wrap_leaf_next (leaf)

    def __del__ (self):
        # Free the 2D p4est internal state
        libp4est.p4est_wrap_destroy (self.wrap)    

        # Call this once at the end of program
        libp4est.p4est_wrap_finalize ()

pdemo = Py4estDemo ()
