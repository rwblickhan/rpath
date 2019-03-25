use rand::prelude::*;
use rayon::prelude::*;

use std::f64::INFINITY;
use std::f64::consts::PI;

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

    fn normal(&self, p0: &Point) -> DirVec {
        nalgebra::Unit::new_unchecked(nalgebra::Vector3::<Scalar>::zeros())
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

fn orthonormal_system(v1: &DirVec) -> (DirVec, DirVec) {
    let v2 = if v1.x.abs() > v1.y.abs() {
        let inv_len = 1.0 / (v1.x * v1.x + v1.z * v1.z).sqrt();
        nalgebra::Unit::new_normalize(nalgebra::Vector3::new(-1.0 * v1.z * inv_len, 0.0, v1.x * inv_len))
    } else {
        let inv_len = 1.0 / (v1.y * v1.y + v1.z * v1.z).sqrt();
        nalgebra::Unit::new_normalize(nalgebra::Vector3::new(0.0, v1.z * inv_len, -v1.y * inv_len))
    };
    (v2, nalgebra::Unit::new_normalize(v1.as_ref().cross(v2.as_ref())))
}

fn sample_hemisphere_uniform(u1: Scalar, u2: Scalar) -> DirVec {
    let r = (1.0 - u1 * u1).sqrt();
    let phi = 2.0 * PI * u2;
    nalgebra::Unit::new_normalize(nalgebra::Vector3::new(phi.cos() * r, phi.sin() * r, u1))
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
    let normal = hit_geometry.normal(&hit_point);

    let emission = hit_material.emission;
    let emission_color = Color::new(emission, emission, emission) * rr_factor;

    match hit_material.mat_type {
        MaterialType::Diffuse => {
            let (rot_x, rot_y) = orthonormal_system(&normal);
            let sampled_dir = sample_hemisphere_uniform(rng.gen(), rng.gen());
            let rotated_dir = nalgebra::Unit::new_normalize(nalgebra::Vector3::new(
                nalgebra::Vector3::new(rot_x.x, rot_y.x, normal.x).dot(&sampled_dir),
                nalgebra::Vector3::new(rot_x.y, rot_y.y, normal.y).dot(&sampled_dir),
                nalgebra::Vector3::new(rot_x.z, rot_y.z, normal.z).dot(&sampled_dir),
            ));
            let out_ray = Ray { origin: hit_point, dir: rotated_dir };
            let cos = out_ray.dir.dot(&normal);
            let indirect_color = trace(&out_ray, scene, depth + 1);
            Some(emission_color + (indirect_color.unwrap_or(Color::zeros()).component_mul(&hit_material.color)) * cos * 0.1 * rr_factor)
        }
        MaterialType::Specular => {
            let cos = ray.dir.dot(&normal);
            let out_ray = Ray {
                origin: hit_point,
                dir: nalgebra::Unit::new_normalize(ray.dir.as_ref() - normal.as_ref() * (cos * 2.0)),
            };
            let indirect_color = trace(&out_ray, scene, depth + 1);
            Some(emission_color + indirect_color.unwrap_or(Color::zeros()) * rr_factor)
        }
        MaterialType::Refractive => {
            // TODO make refractive index configurable
            let mut n: f64 = 1.5;
            let mut mut_normal = normal;
            let r0 = ((1.0 - n) / (1.0 + n)).powi(2);
            if mut_normal.dot(ray.dir.as_ref()) > 0.0 {
                mut_normal = nalgebra::Unit::new_normalize(mut_normal.into_inner() * -1.0);
                n = 1.0 / n;
            }
            n = 1.0 / n;
            let cos1 = mut_normal.dot(ray.dir.as_ref()) * -1.0;
            let cos2 = 1.0 - n * n * (1.0 - cos1 * cos1);
            let r_prob = (r0 * (1.0 - r0)) * (1.0 - cos1).powi(5);
            let out_ray = Ray {
                origin: hit_point,
                dir: if cos2 > 0.0 && rng.gen::<f64>() > r_prob {
                    nalgebra::Unit::new_normalize((ray.dir.as_ref() * n) + (mut_normal.as_ref() * (n * cos1 - cos2.sqrt())))
                } else {
                    nalgebra::Unit::new_normalize(ray.dir.as_ref() + mut_normal.as_ref() * (cos1 * 2.0))
                },
            };
            let indirect_color = trace(&out_ray, scene, depth + 1);
            Some(emission_color + indirect_color.unwrap_or(Color::zeros()) * 1.15 * rr_factor)
        }
    }
}

fn main() {
    println!("Hello, world!");
}