use rand::prelude::*;

type Scalar = f64;
type Point = nalgebra::Point3<Scalar>;
type DirVec = nalgebra::Unit<nalgebra::Vector3<Scalar>>;
type Color = nalgebra::Vector3<Scalar>;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Ray {
    origin: Point,
    dir: DirVec,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaterialType {
    Diffuse,
    Specular,
    Refractive,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Material {
    color: Color,
    emission: Scalar,
    mat_type: MaterialType,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Geometry {
    Plane {
        normal: DirVec,
        offset: Scalar,
    },
    Sphere {
        center: Point,
        radius: Scalar,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Object {
    geometry: Geometry,
    material: Material,
}

#[derive(Clone, Default, Debug, PartialEq)]
struct Scene {
    objects: Vec<Object>,
}

fn trace(ray: &Ray, scene: &Scene, depth: u32) -> Option<Color> {
    let mut rng = rand::thread_rng();

    // Russian roulette stopping
    let rr_factor = if depth >= 5 {
        let stop_prob = 0.1;
        if rng.gen::<f64>() <= stop_prob { return None; } else { 1.0 / 1.0 - stop_prob }
    } else { 1.0 };

    None
}

fn main() {
    println!("Hello, world!");
}