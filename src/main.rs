use rand::prelude::*;
use rayon::prelude::*;

use std::f64::INFINITY;

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

impl Geometry {
    fn intersect(&self, ray: &Ray) -> Scalar {
        match self {
            &Geometry::Plane { normal, offset } => {
                let d0 = normal.dot(ray.dir.as_ref());
                if d0 != 0.0 {
                    let t = -1.0 * (normal.dot(&ray.origin.coords) + offset) / d0;
                    if t > 1e-6 { t } else { 0.0 }
                } else { 0.0 }
            }
            &Geometry::Sphere { center, radius } => {
                let b = ((ray.origin - center) * 2.0).dot(ray.dir.as_ref());
                let c = (ray.origin - center).dot(&(ray.origin - center)) - (radius * radius);
                let disc_sq = b * b - 4.0 * c;
                if disc_sq < 0.0 {
                    return 0.0;
                }
                let disc = disc_sq.sqrt();
                let sol1 = -b + disc;
                let sol2 = -b - disc;
                if sol2 > 1e-6 { sol2 / 2.0 } else { if sol1 > 1e-6 { sol1 / 2.0 } else { 0.0 } }
            }
        }
    }
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

impl Scene {
    fn intersect(&self, ray: &Ray) -> (Scalar, Option<&Object>) {
        self.objects.par_iter()
            .fold(|| (INFINITY, None),
                  |(t, obj), next_obj| {
                      let next_obj_t = next_obj.geometry.intersect(ray);
                      if next_obj_t > 1e-6 && next_obj_t < t {
                          (next_obj_t, Some(next_obj))
                      } else { (t, obj) }
                  })
            .reduce(|| (INFINITY, None),
                    |(t, obj), (next_t, next_obj)| {
                        if next_t > 1e-6 && next_t < t {
                            (next_t, next_obj)
                        } else { (t, obj) }
                    })
    }
}

fn trace(ray: &Ray, scene: &Scene, depth: u32) -> Option<Color> {
    let mut rng = rand::thread_rng();

    // Russian roulette stopping
    let rr_factor = if depth >= 5 {
        let stop_prob = 0.1;
        if rng.gen::<f64>() <= stop_prob { return None; } else { 1.0 / 1.0 - stop_prob }
    } else { 1.0 };

    // Find ray intersection
    let (t, object) = scene.intersect(ray);
    if object.is_none() { return None; }
    let hit_point = &ray.origin + t * ray.dir.into_inner();
    let hit_material = object.unwrap().material;
    let hit_geometry = object.unwrap().geometry;

    None
}

fn main() {
    println!("Hello, world!");
}