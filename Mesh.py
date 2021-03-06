#Meshing the airfoil using Pointwise
#Coded by Alfiyandy Hariansyah
#Tohoku University
#3/2/2021
#####################################################################################################
from LoadVars import pw_port
from pointwise import GlyphClient
from pointwise.glyphapi import *

#Establishing a pointwise environment
glf = GlyphClient(port=pw_port)
pw = glf.get_glyphapi()

class AirfoilMesh():
	"""An airfoil object that can do the mesh process"""
	def __init__(self, file_in, file_out,
				 con_dimension,
				 le_spacing, te_spacing,
				 solver, dim,
				 mesh_type):

		"""It contains parameters to be used"""
		# file_in = 'D:/my_project/transonic-airfoil-optim/Designs/gen_' + file_in
		# file_out = 'D:/my_project/transonic-airfoil-optim/Solutions/gen_' + file_out
		file_in = 'D:/my_project/transonic-airfoil-optim/Designs/baseline/' + file_in
		file_out = 'D:/my_project/transonic-airfoil-optim/Grid_Convergence_Study/baseline/ogrid-euler/' + file_out

		self.file_in = file_in
		self.file_out = file_out
		self.con_dimension = con_dimension
		self.le_spacing = le_spacing
		self.te_spacing = te_spacing
		self.solver = solver
		self.dim = dim
		if mesh_type in ['OGRID_STRUCTURED','CGRID_STRUCTURED']:
			self.mesh_type = 'STRUCTURED'
		elif mesh_type == 'OGRID_UNSTRUCTURED':
			self.mesh_type = 'UNSTRUCTURED'

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

	def create_airfoil_connectors(self):
		"""Creating connectors"""
		pw.Connector.setCalculateDimensionMaximum(2000)
		pw.Connector.setDefault('Dimension',self.con_dimension)
		self.curves = pw.Database.getAll()
		self.connectors = pw.Connector.createOnDatabase(self.curves,
			parametricConnectors='Aligned', merge=0,
			type=self.mesh_type)
		pw.Application.markUndoLevel('Connectors On DB Entities')

	def create_connector_spline(self,point1,point2):
		"""Create a connector by two points"""
		with pw.Application.begin('Create') as creator:
			seg1 = pw.SegmentSpline()
			seg1.addPoint(point1)
			seg1.addPoint(point2)

			con = pw.Connector()
			con.addSegment(seg1)
			creator.end()

	def create_connector_circle(self,point1,point2,angle):
		"""Create a circle connector by two points and an angle"""
		with pw.Application.begin('Create') as creator:
			seg1 = pw.SegmentCircle()
			seg1.addPoint(point1)
			seg1.addPoint(point2)
			seg1.setAngle(angle)

			con = pw.Connector()
			con.addSegment(seg1)
			creator.end()

	def modify_connectors(self):
		"""To display points in the connectors (optional)"""
		connector_1 = pw.GridEntity.getByName('con-1')
		connector_2 = pw.GridEntity.getByName('con-2')
		connector_1.setRenderAttribute('PointMode','All')
		connector_2.setRenderAttribute('PointMode','All')
		pw.Application.markUndoLevel('Modify Entity Display')

	def change_airfoil_spacing(self):
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

	def change_connector_spacing(self,con_name,spacing,where):
		"""Change Leading Edge and Trailing Edge Spacing"""
		con = pw.GridEntity.getByName(con_name)	
		with pw.Application.begin('Modify',con) as modifier:
			if where == 'begin':
				con.getDistribution(1).setBeginSpacing(spacing)
			if where == 'end':
				con.getDistribution(1).setEndSpacing(spacing)
			if where == 'begin_and_end':
				con.getDistribution(1).setBeginSpacing(spacing)
				con.getDistribution(1).setEndSpacing(spacing)
			modifier.end()
		pw.Application.markUndoLevel('Change Spacing')

	def pw_export(self,domains):
		"""Exporting mesh files, e.g. SU2 mesh"""
		#domains is a list of domains
		dom_export = pw.Entity.sort(domains)
		with pw.Application.begin('CaeExport',dom_export) as exporter:
			exporter.initialize(self.file_out,
				strict=True, type='CAE')
			exporter.setAttribute('FilePrecision','Double')
			exporter.verify()
			exporter.write()
			exporter.end()

	def cgrid_structured(self,farfield_radius,step_dim,first_spacing):
		"""Completing the routines for c-grid topology structured mesh"""
		self.pw_reset()
		self.pw_reset()
		self.modify_tolerances()
		self.set_CAE_solver()
		self.pw_import()
		self.create_airfoil_connectors()
		self.change_airfoil_spacing()

		wake_dist = farfield_radius

		pw.Connector.setDefault('Dimension',2*step_dim)
		self.create_connector_spline([1,0,0],[wake_dist,0,0])
		self.change_connector_spacing('con-3',self.te_spacing,'begin')

		pw.Connector.setDefault('Dimension',step_dim)
		pw.Connector.setDefault('BeginSpacing',first_spacing)
		self.create_connector_spline([wake_dist,0,0],[wake_dist,farfield_radius,0])
		self.create_connector_spline([wake_dist,0,0],[wake_dist,-farfield_radius,0])
		self.create_connector_spline([1,0,0],[1,farfield_radius,0])
		self.create_connector_spline([1,0,0],[1,-farfield_radius,0])

		pw.Connector.setDefault('Dimension',2*step_dim)
		pw.Connector.setDefault('BeginSpacing',0)
		self.create_connector_spline([1,farfield_radius,0],[wake_dist,farfield_radius,0])
		self.create_connector_spline([1,-farfield_radius,0],[wake_dist,-farfield_radius,0])
		self.change_connector_spacing('con-8',self.te_spacing,'begin')
		self.change_connector_spacing('con-9',self.te_spacing,'begin')

		pw.Connector.setDefault('Dimension',2*self.con_dimension-1)
		self.create_connector_circle([1,farfield_radius,0],[1,-farfield_radius,0],180)
		self.change_connector_spacing('con-10',self.te_spacing,'begin_and_end')

		#Get grid entity
		con1 = pw.GridEntity.getByName('con-1')
		con2 = pw.GridEntity.getByName('con-2')
		con3 = pw.GridEntity.getByName('con-3')
		con4 = pw.GridEntity.getByName('con-4')
		con5 = pw.GridEntity.getByName('con-5')
		con6 = pw.GridEntity.getByName('con-6')
		con7 = pw.GridEntity.getByName('con-7')
		con8 = pw.GridEntity.getByName('con-8')
		con9 = pw.GridEntity.getByName('con-9')
		con10 = pw.GridEntity.getByName('con-10')

		#Create domains
		with pw.Application.begin('Create') as creator:
			edge_1 = pw.Edge.create()
			edge_1.addConnector(con6)
			edge_2 = pw.Edge.create()
			edge_2.addConnector(con10)
			edge_3 = pw.Edge.create()
			edge_3.addConnector(con7)
			edge_4 = pw.Edge.create()
			edge_4.addConnector(con1)
			edge_4.addConnector(con2)

			#for baseline (switch!)
			# edge_4.addConnector(con2)
			# edge_4.addConnector(con1)

			dom = pw.DomainStructured.create()
			dom.addEdge(edge_1)
			dom.addEdge(edge_2)
			dom.addEdge(edge_3)
			dom.addEdge(edge_4)
			creator.end()
		dom2 = pw.DomainStructured.createFromConnectors([con3, con4, con6, con8])
		dom3 = pw.DomainStructured.createFromConnectors([con3, con5, con7, con9])

		#Set the boundary conditions
		dom1 = pw.GridEntity.getByName('dom-1')
		dom2 = pw.GridEntity.getByName('dom-2')
		dom3 = pw.GridEntity.getByName('dom-3')

		bc_airfoil = pw.BoundaryCondition.create()
		bc_airfoil.setName('airfoil')
		airfoil_doms = [[dom1, con1],[dom1, con2]]
		bc_airfoil.apply(airfoil_doms)

		bc_farfield = pw.BoundaryCondition.create()
		bc_farfield.setName('farfield')
		farfield_doms = [[dom1,con10],[dom2,con8],[dom2,con4],[dom3,con9],[dom3,con5]]
		bc_farfield.apply(farfield_doms)

		#Export the mesh files
		self.pw_export([dom1, dom2, dom3])

	def ogrid_structured(self, growth_factor=1.2, initial_stepsize=0.001, step=100):
		"""Completing the routines for structured mesh"""
		self.pw_reset()
		self.modify_tolerances()
		self.set_CAE_solver()
		self.pw_import()
		self.create_airfoil_connectors()
		#self.modify_connectors()
		self.change_airfoil_spacing()

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

		#Export the mesh files
		self.pw_export()

	def ogrid_unstructured(self, algorithm,
					 size_field_decay,
					 farfield_radius,
					 farfield_dim,):
		"""Completing the routines for unstructured mesh"""
		self.pw_reset()
		self.modify_tolerances()
		self.set_CAE_solver()
		self.pw_import()
		self.create_airfoil_connectors()
		#self.modify_connectors()
		self.change_airfoil_spacing()

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
		self.pw_export([dom_1])