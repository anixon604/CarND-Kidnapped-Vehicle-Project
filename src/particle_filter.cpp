/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *			Student work: Anthony Nixon
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	//  Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	num_particles = 25;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	default_random_engine gen;

	for(int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;

	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// xf = x0 + v/yaw_rate [sin(theta + yaw_rate(dt)) - sin(theta)]
	// yf = y0 + v/yaw_rate [cos(theta) - cos(theta + yaw_rate(dt))]
	// heading = theta + yaw_rate(dt)

	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	// sub formulas
	double v_div_y = velocity / yaw_rate;
	double y_mul_t = yaw_rate * delta_t;

	default_random_engine gen;

	for(Particle& p : particles) {
		// add measurements

		// yaw rate is close to zero
		if(fabs(yaw_rate) < 0.000001) {
			p.x = p.x + velocity * delta_t * cos(p.theta);
			p.y = p.y + velocity * delta_t * sin(p.theta);
		} else { // yaw rate not equal to zero
			p.x = p.x + v_div_y * (sin(p.theta + y_mul_t) - sin(p.theta));
			p.y = p.y + v_div_y * (cos(p.theta) - cos(p.theta + y_mul_t));
			p.theta = p.theta + y_mul_t;
		}

		// distribution with mean of zero gets added to coord as noise
		normal_distribution<double> dist_x(0, std_x);
		normal_distribution<double> dist_y(0, std_y);
		normal_distribution<double> dist_theta(0, std_theta);

		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);
	}
}

/**
	* helper function that transforms the landmarks MAP coordinates (particles and landmarks)
	* to VEHICLE coordinates
	* - Map coordinates will be translated by particle position
	*	- Map coordinates will be rotated to partical heading angle
	* @ param observations Vector of landmark observations
  */
std::vector<LandmarkObs> transformCoords(double sensor_range, Particle p, Map map_landmarks) {

	std::vector<LandmarkObs> predicted;

	double partX = p.x;
	double partY = p.y;
	double partHeading = p.theta;
	double landX, landY;

	for(int i = 0; i < map_landmarks.landmark_list.size(); i++) {
		Map::single_landmark_s landmark = map_landmarks.landmark_list[i];
		landX = landmark.x_f;
		landY = landmark.y_f;

		// yaw now in opposite direction subtract 180 degrees
		double cos_theta = cos(partHeading - M_PI / 2);
		double sin_theta = sin(partHeading - M_PI / 2);

		LandmarkObs l;
		l.id = landmark.id_i;
		l.x = -(landX-partX)*sin_theta+(landY-partY)*cos_theta;
		l.y = -(landX-partX)*cos_theta-(landY-partY)*sin_theta;

		// sensor_range filter
	  if(dist(0, 0, l.x, l.y) <= sensor_range) predicted.push_back(l);
	}

	return predicted;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for(LandmarkObs& obsLandmark : observations) {
		double minDistance = std::numeric_limits<double>::max();
		double currDist = 0;

		for(LandmarkObs predLandmark : predicted) {
			currDist = dist(obsLandmark.x, obsLandmark.y, predLandmark.x, predLandmark.y);
			if(currDist < minDistance) {
				//cout <<  "Distance " << minDistance << " " << obsLandmark.id << " " << predLandmark.id << " ID" << endl;
				obsLandmark.id = predLandmark.id;
				minDistance = currDist;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	//   Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// vector<LandmarkObs> predicted - a simulated observation of each landmark relative to the particle.
	//cout << "updateWeights started" << particles.size() << weights.size() << endl;
	std::vector<LandmarkObs> predicted;

	for(int i = 0; i < num_particles; i++) {
			predicted = transformCoords(sensor_range, particles[i], map_landmarks);
			dataAssociation(predicted, observations);

			// BEGIN calculate Multivariate_normal_distribution
			/* Covariance Matrix E:
					| variance_x  0 |
					| 0  variance_y |

				 Inverse of Covariance Matrix E:
				 1/[variance_x*variance_y] * | variance_y   0 |
				 									 					 | 0   variance_x |
			*/
			double var_x = std_landmark[0]*std_landmark[0];
			double var_y = std_landmark[1]*std_landmark[1];
			double invCovX = 1/var_x;
			double invCovY = 1/var_y;
			double delta_x, delta_y, ximu2, yimu2;

			double weight = 1.0;

			for(LandmarkObs obs : observations) {

				// find associated prediction id / observations id for calculations
				LandmarkObs p;
				for(LandmarkObs landmark : predicted) {
					if(obs.id == landmark.id) {
						p = landmark;
						break;
					}
				}

				// update subformulas
				delta_x = obs.x - p.x;
				delta_y = obs.y - p.y;
				ximu2 = delta_x*delta_x;
				yimu2 = delta_y*delta_y;

				weight *= exp((-1.0/2)*(ximu2*invCovX + yimu2*invCovY))
													*(1/(2*M_PI*std_landmark[0]*std_landmark[1]));
				// sqrt[|2PI*E|] = sqrt[determinant of 2PI * E]
				//float denominator = sqrt(1/((2*M_PI*invCovX)*(2*M_PI*invCovY)));

				//END calculate Multivariate_normal_distribution

				// DEBUG PRINTOUTS

				// cout << "\nVar x " << var_x << endl;
				// cout << "Var y " << var_y << endl;
				// cout << "invCovX " << invCovX << endl;
				// cout << "invCovY " << invCovY << endl;
				// cout << "delta_x " << delta_x << endl;
				// cout << "delta_y " << delta_y << endl;
				// cout << "ximu2 " << ximu2 << endl;
				// cout << "yimu2 " << yimu2 << endl;
				// cout << "numerator " << numerator << endl;
				// cout << "denominator " << denominator << endl;
				// cout << "weight " << weight << "\n" << endl;
				// cout << "ximu2 " << ximu2 << " yimu2 " << yimu2 << endl;
				// cout << numerator << " " << denominator <<  " " << weight << endl;
			}

			particles[i].weight = weight; // assign new weight to particle.
			weights[i] = weight; // update weights vector

	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::default_random_engine gen;
	gen.seed(824);

	std::discrete_distribution<int> distribution(weights.begin(), weights.end());

	std::vector<Particle> newer_part;
	newer_part.resize(num_particles);

	for(int i=0; i < num_particles; i++) {
		newer_part[i] = particles[distribution(gen)];
	}

	particles = newer_part;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
