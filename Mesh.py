#Meshing the airfoil using Pointwise
#Coded by Alfiyandy Hariansyah
#Tohoku University
#2/3/2021
#####################################################################################################
from pointwise import GlyphClient
from pointwise.glyphapi import *

#Establishing a pointwise environment
glf = GlyphClient(port=2807)
pw = glf.get_glyphapi()

class AirfoilMesh():
	"""An airfoil object that can do the mesh process"""
	def __init__(self, file_in, file_out,
				 con_dimension,
				 le_spacing, te_spacing,
				 solver, dim,
				 mesh_type):

		"""It contains parameters to be used"""
		file_in = 'D:/my_project/transonic-airfoil-optim/Designs/initial_samples/bspline_points/' + file_in
		file_out = 'D:/my_project/transonic-airfoil-optim/Meshes/gen_1/' + file_out

		self.file_in = file_in
		self.file_out = file_out
		self.con_dimension = con_dimension
		self.le_spacing = le_spacing
		self.te_spacing = te_spacing
		self.solver = solver
		self.dim = dim
		self.mesh_type = mesh_type

		self.curves = []
		self.connectors = []

	def pw_reset(self):
		"""Resetting pointwise"""
		pw.Application.setUndoMaximumLevels(5)
		pw.Application.reset()
		pw.Application.markUndoLevel('Journal Reset')
		pw.Application.clearModified()		

	def modify_tolerances(self):
		"""Modifying tolerances, so that it matches the real values"""
		pw.Database.setModelSize(1)
		pw.Grid.setNodeTolerance(9.9999999999999995e-08)
		pw.Grid.setConnectorTolerance(9.9999999999999995e-08)
		pw.Grid.setGridPointTolerance(1e-10)
		pw.Application.markUndoLevel('Modify Tolerances')

	def set_CAE_solver(self):
		"""Setting the CAE solver, e.g. SU2"""
		pw.Application.setCAESolver(self.solver, self.dim)

	def pw_import(self):
		"""Importing Database"""
		with pw.Application.begin('DatabaseImport') as importer:
			importer.initialize(self.file_in, strict=True)
			importer.read()
			importer.convert()
			importer.end()
		pw.Application.markUndoLevel('Import Database')

	def create_connectors(self):
		"""Creating connectors"""
		pw.Connector.setDefault('Dimension',self.con_dimension)
		self.curves = pw.Database.getAll()
		self.connectors = pw.Connector.createOnDatabase(self.curves,
			parametricConnectors='Aligned', merge=0,
			type=self.mesh_type)
		pw.Application.markUndoLevel('Connectors On DB Entities')

	def modify_connectors(self):
		"""To display points in the connectors (optional)"""
		connector_1 = pw.GridEntity.getByName('con-1')
		connector_2 = pw.GridEntity.getByName('con-2')
		connector_1.setRenderAttribute('PointMode','All')
		connector_2.setRenderAttribute('PointMode','All')
		pw.Application.markUndoLevel('Modify Entity Display')

	def change_spacing(self):
		"""Change Leading Edge and Trailing Edge Spacing"""
		connector_1 = pw.GridEntity.getByName('con-1')
		connector_2 = pw.GridEntity.getByName('con-2')		
		with pw.Application.begin('Modify',self.connectors) as modifier:
			connector_1.getDistribution(1).setBeginSpacing(self.le_spacing)
			connector_1.getDistribution(1).setEndSpacing(self.te_spacing)
			connector_2.getDistribution(1).setBeginSpacing(self.te_spacing)
			connector_2.getDistribution(1).setEndSpacing(self.le_spacing)
			modifier.end()
		pw.Application.markUndoLevel('Change Spacing')

	def set_bc(self):
		"""Setting the boundary conditions"""
		connector_1 = pw.GridEntity.getByName('con-1')
		dom_1 = pw.GridEntity.getByName('dom-1')
		connector_2 = pw.GridEntity.getByName('con-2')
		connector_3 = pw.GridEntity.getByName('con-3')
		connector_4 = pw.GridEntity.getByName('con-4')

		bc_airfoil = pw.BoundaryCondition.create()
		bc_airfoil.setName('airfoil')
		airfoil_doms = [[dom_1, connector_1],[dom_1, connector_2]]
		bc_airfoil.apply(airfoil_doms)

		bc_farfield = pw.BoundaryCondition.create()
		bc_farfield.setName('farfield')
		farfield_doms = [[dom_1,connector_4]]
		bc_farfield.apply(farfield_doms)

	def pw_export(self):
		"""Exporting mesh files, e.g. SU2 mesh"""
		dom_1 = pw.GridEntity.getByName('dom-1')
		dom_export = pw.Entity.sort([dom_1])
		with pw.Application.begin('CaeExport',dom_export) as exporter:
			exporter.initialize(self.file_out,
				strict=True, type='CAE')
			exporter.setAttribute('FilePrecision','Double')
			exporter.verify()
			exporter.write()
			exporter.end()

	def structured(self, growth_factor=1.2, initial_stepsize=0.001, step=100):
		"""Completing the routines for structured mesh"""
		self.pw_reset()
		self.modify_tolerances()
		self.set_CAE_solver()
		self.pw_import()
		self.create_connectors()
		#self.modify_connectors()
		self.change_spacing()

		#Extrude normal
		with pw.Application.begin('Create') as creator:
			inner_edge = pw.Edge.createFromConnectors(self.connectors)
			dom = pw.DomainStructured.create()
			dom.addEdge(inner_edge)
			creator.end()

		with pw.Application.begin('ExtrusionSolver',dom) as ExtrusionSolver:
			ExtrusionSolver.setKeepFailingStep(True)
			dom.setExtrusionSolverAttribute('SpacingGrowthFactor',growth_factor)
			dom.setExtrusionSolverAttribute('NormalInitialStepSize',initial_stepsize)
			ExtrusionSolver.run(step)
			ExtrusionSolver.end()

		pw.Application.markUndoLevel('Extrude normal')

		#Set the boundary conditions
		self.set_bc()

		#Export the mesh files
		self.pw_export()

	def unstructured(self, algorithm,
					 size_field_decay,
					 farfield_radius,
					 farfield_dim,):
		"""Completing the routines for unstructured mesh"""
		self.pw_reset()
		self.modify_tolerances()
		self.set_CAE_solver()
		self.pw_import()
		self.create_connectors()
		#self.modify_connectors()
		self.change_spacing()

		#Creating farfield connector
		pw.Connector.setDefault('Dimension',farfield_dim)
		with pw.Application.begin('Create') as creator:
			seg1 = pw.SegmentCircle()
			seg1.addPoint([farfield_radius, 0, 0])
			seg1.addPoint([0, 0, 0])
			seg1.setEndAngle(360)

			farfield = pw.Connector()
			farfield.addSegment(seg1)

			creator.end()
		pw.Application.markUndoLevel('Create Connector')

		#Select algorithm for Unstructured mesh
		pw.DomainUnstructured.setDefault('Algorithm', algorithm)

		#Set boundary decay
		pw.GridEntity.setDefault('SizeFieldDecay', size_field_decay)

		#Creating unstructured domain
		connector_1 = pw.GridEntity.getByName('con-1')
		connector_2 = pw.GridEntity.getByName('con-2')
		connector_3 = pw.GridEntity.getByName('con-3')
		with pw.Application.begin('Create') as creator:
			edge_1 = pw.Edge.create()
			edge_1.addConnector(connector_3)
			edge_2 = pw.Edge.create()
			edge_2.addConnector(connector_1)
			edge_2.addConnector(connector_2)

			dom = pw.DomainUnstructured.create()
			dom.addEdge(edge_1)
			dom.addEdge(edge_2)
			creator.end()
		pw.Application.markUndoLevel('Assemble Domain')
		

		#Set the boundary conditions
		connector_1 = pw.GridEntity.getByName('con-1')
		dom_1 = pw.GridEntity.getByName('dom-1')
		connector_2 = pw.GridEntity.getByName('con-2')
		connector_3 = pw.GridEntity.getByName('con-3')

		bc_airfoil = pw.BoundaryCondition.create()
		bc_airfoil.setName('airfoil')
		airfoil_doms = [[dom_1, connector_1],[dom_1, connector_2]]
		bc_airfoil.apply(airfoil_doms)

		bc_farfield = pw.BoundaryCondition.create()
		bc_farfield.setName('farfield')
		farfield_doms = [[dom_1,connector_3]]
		bc_farfield.apply(farfield_doms)

		#Exporting the mesh files
		self.pw_export()




