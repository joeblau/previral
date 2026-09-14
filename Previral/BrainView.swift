import SceneKit
import SwiftUI

/// Interactive cortical anatomy with a persistent scene and camera. Only the
/// color source changes during playback; mesh topology and lighting are reused.
struct BrainView: NSViewRepresentable {
    var mesh: BrainMesh?
    var vertexColors: [SIMD4<Float>]?
    var inflated = false
    var resetVersion = 0

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.backgroundColor = NSColor(white: 0.035, alpha: 1)
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = false
        view.antialiasingMode = .multisampling4X
        view.defaultCameraController.inertiaEnabled = true
        return view
    }

    func updateNSView(_ view: SCNView, context: Context) {
        guard let mesh, mesh.vertexCount > 0 else {
            view.scene = nil
            context.coordinator.reset()
            return
        }
        let coordinator = context.coordinator
        let colors = vertexColors?.count == mesh.vertexCount ? vertexColors! : mesh.anatomyColors
        let newMesh = coordinator.positions != mesh.pialPositions
        let surfaceChanged = coordinator.inflated != inflated
        if newMesh || view.scene == nil {
            let setup = Self.makeScene(mesh: mesh, colors: colors, inflated: inflated)
            view.scene = setup.scene
            view.pointOfView = setup.camera
            coordinator.node = setup.brain
            coordinator.camera = setup.camera
            coordinator.positions = mesh.pialPositions
            view.defaultCameraController.target = setup.brain.boundingSphere.center
        } else if surfaceChanged || coordinator.colors != colors {
            let geometry: SCNGeometry
            if surfaceChanged {
                geometry = Self.makeGeometry(mesh: mesh, colors: colors, inflated: inflated)
            } else if let previous = coordinator.node?.geometry {
                geometry = SCNGeometry(sources: previous.sources(for: .vertex)
                    + previous.sources(for: .normal) + [Self.colorSource(colors)], elements: previous.elements)
                geometry.materials = previous.materials
                geometry.subdivisionLevel = previous.subdivisionLevel
            } else { return }
            SCNTransaction.begin()
            SCNTransaction.disableActions = true
            coordinator.node?.geometry = geometry
            SCNTransaction.commit()
        }
        if newMesh || surfaceChanged || coordinator.resetVersion != resetVersion {
            if let brain = coordinator.node, let camera = coordinator.camera {
                Self.frameCamera(camera, brain: brain)
                view.pointOfView = camera
                view.defaultCameraController.stopInertia()
                view.defaultCameraController.target = brain.convertPosition(brain.boundingSphere.center, to: nil)
            }
        }
        coordinator.inflated = inflated
        coordinator.colors = colors
        coordinator.resetVersion = resetVersion
    }

    static func makeScene(mesh: BrainMesh, colors: [SIMD4<Float>], inflated: Bool)
        -> (scene: SCNScene, brain: SCNNode, camera: SCNNode) {
        let scene = SCNScene()
        scene.background.contents = NSColor(white: 0.035, alpha: 1)
        let brain = SCNNode(geometry: makeGeometry(mesh: mesh, colors: colors, inflated: inflated))
        // FreeSurfer RAS has Z up. SceneKit has Y up.
        brain.eulerAngles.x = -.pi / 2
        scene.rootNode.addChildNode(brain)
        let camera = SCNNode()
        camera.camera = SCNCamera()
        camera.camera?.usesOrthographicProjection = true
        camera.camera?.zNear = 1
        camera.camera?.zFar = 2000
        camera.camera?.wantsHDR = false
        camera.camera?.screenSpaceAmbientOcclusionIntensity = 0.65
        camera.camera?.screenSpaceAmbientOcclusionRadius = 4
        camera.camera?.screenSpaceAmbientOcclusionBias = 0.3
        scene.rootNode.addChildNode(camera)
        frameCamera(camera, brain: brain)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 200
        ambient.light?.color = NSColor.white
        scene.rootNode.addChildNode(ambient)
        // Camera-relative soft studio lights keep the folds legible while orbiting.
        for (position, intensity) in [(SCNVector3(-180, 220, 160), CGFloat(900)),
                                      (SCNVector3(180, 40, 80), CGFloat(250))] {
            let light = SCNNode()
            light.light = SCNLight()
            light.light?.type = .omni
            light.light?.intensity = intensity
            light.position = position
            camera.addChildNode(light)
        }
        return (scene, brain, camera)
    }

    static func frameCamera(_ camera: SCNNode, brain: SCNNode) {
        let center = brain.convertPosition(brain.boundingSphere.center, to: nil)
        let (lo, hi) = brain.boundingBox
        let extent = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
        camera.camera?.orthographicScale = Double(extent) * 0.62
        camera.position = SCNVector3(center.x + extent * 0.16, center.y + extent * 1.25, center.z + extent * 1.65)
        camera.look(at: center)
    }

    static func makeGeometry(mesh: BrainMesh, colors: [SIMD4<Float>], inflated: Bool) -> SCNGeometry {
        func source(_ floats: [Float], semantic: SCNGeometrySource.Semantic) -> SCNGeometrySource {
            SCNGeometrySource(data: floats.withUnsafeBufferPointer { Data(buffer: $0) },
                              semantic: semantic, vectorCount: mesh.vertexCount,
                              usesFloatComponents: true, componentsPerVector: 3,
                              bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        }
        let element = SCNGeometryElement(
            data: mesh.faces.withUnsafeBufferPointer { Data(buffer: $0) },
            primitiveType: .triangles, primitiveCount: mesh.faces.count / 3, bytesPerIndex: 4)
        let geometry = SCNGeometry(sources: [
            source(inflated ? mesh.inflatedPositions : mesh.pialPositions, semantic: .vertex),
            source(inflated ? mesh.inflatedNormals : mesh.normals, semantic: .normal),
            colorSource(colors),
        ], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .lambert
        material.diffuse.contents = NSColor.white
        material.ambient.contents = NSColor(white: 0.08, alpha: 1)
        material.locksAmbientWithDiffuse = false
        material.isDoubleSided = false
        geometry.materials = [material]
        // Render-only subdivision softens the silhouette without changing the
        // model's 20,484 prediction vertices or their anatomical registration.
        geometry.subdivisionLevel = 1
        return geometry
    }

    private static func colorSource(_ colors: [SIMD4<Float>]) -> SCNGeometrySource {
        // Geometry color attributes are linear RGB; our shared UI palette is
        // sRGB. Skipping this conversion washes reds into pale orange.
        func linear(_ x: Float) -> Float {
            x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4)
        }
        let linearColors = colors.map { SIMD4(linear($0.x), linear($0.y), linear($0.z), $0.w) }
        return SCNGeometrySource(data: linearColors.withUnsafeBufferPointer { Data(buffer: $0) },
                          semantic: .color, vectorCount: colors.count, usesFloatComponents: true,
                          componentsPerVector: 4, bytesPerComponent: 4, dataOffset: 0,
                          dataStride: MemoryLayout<SIMD4<Float>>.stride)
    }

    final class Coordinator {
        var positions: [Float] = []
        var colors: [SIMD4<Float>] = []
        var inflated = false
        var resetVersion = 0
        var node: SCNNode?
        var camera: SCNNode?
        func reset() { positions = []; colors = []; node = nil; camera = nil }
    }
}
