#!/usr/bin/env python3
"""
FounderMatch RAG Data Generation Script
Generates synthetic founder data and precomputes embeddings for fast search.
"""

import json
import csv
import uuid
import random
from faker import Faker
import os
from typing import List, Dict
import sys
import argparse
from embeddings_helper import HuggingFaceEmbeddings

# Resolve paths relative to this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Set fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)

fake = Faker()
fake_in = Faker('en_IN')

def ensure_data_dir() -> None:
	os.makedirs(DATA_DIR, exist_ok=True)

class DataGenerator:
	def __init__(self):
		self.roles = ['Founder', 'Co-founder', 'Engineer', 'PM', 'Investor', 'Other']
		self.stages = ['none', 'pre-seed', 'seed', 'series A', 'growth']
		
		# Industry keywords and ideas
		self.industry_keywords = {
			'healthtech': ['telemedicine', 'digital health', 'medical devices', 'health analytics', 'wellness'],
			'fintech': ['payments', 'lending', 'cryptocurrency', 'banking', 'wealth management'],
			'ai': ['machine learning', 'computer vision', 'nlp', 'automation', 'data analytics'],
			'saas': ['enterprise software', 'productivity tools', 'crm', 'collaboration', 'workflow'],
			'ecommerce': ['marketplace', 'retail tech', 'supply chain', 'logistics', 'consumer goods'],
			'edtech': ['online learning', 'educational software', 'skill training', 'certification'],
			'mobility': ['transportation', 'autonomous vehicles', 'logistics', 'delivery'],
			'climate': ['clean energy', 'sustainability', 'carbon tracking', 'green technology'],
			'web3': ['blockchain', 'defi', 'nft', 'cryptocurrency', 'decentralized'],
			'biotech': ['drug discovery', 'genomics', 'medical research', 'pharmaceuticals']
		}
		
		# Location diversity
		self.locations = [
			'San Francisco, USA', 'New York, USA', 'Los Angeles, USA', 'Austin, USA', 'Seattle, USA',
			'London, UK', 'Berlin, Germany', 'Paris, France', 'Amsterdam, Netherlands',
			'Singapore', 'Tel Aviv, Israel', 'Sydney, Australia', 'Toronto, Canada',
			'Bangalore, India', 'Mumbai, India', 'Beijing, China', 'Shanghai, China',
			'Tokyo, Japan', 'Seoul, South Korea', 'São Paulo, Brazil', 'Mexico City, Mexico',
			'Dubai, UAE', 'Stockholm, Sweden', 'Copenhagen, Denmark', 'Zurich, Switzerland'
		]
		
		# Background templates for diversity
		self.background_templates = [
			"Former {prev_role} at {prev_company} with {years} years in {industry}. {achievement}. {skills}.",
			"Ex-{prev_role} from {prev_company}, specialized in {specialty}. {achievement}. {current_focus}.",
			"Serial entrepreneur with {years} years building {industry} companies. {achievement}. {expertise}.",
			"{education} graduate turned {current_role}. {achievement}. {passion}.",
			"Previously led {team_type} at {prev_company}. {achievement}. {vision}."
		]
	
	def generate_founder_data(self, num_founders: int = 700) -> List[Dict]:
		"""Generate synthetic founder data"""
		founders = []
		
		print(f"Generating {num_founders} founder profiles...")
		
		for i in range(num_founders):
			if i % 100 == 0:
				print(f"Generated {i}/{num_founders} profiles...")
			
			# Basic info
			founder_id = str(uuid.uuid4())
			# Prefer Indian names for realism when requested via env flag or default mix
			if os.getenv('USE_INDIAN_NAMES', '1') == '1':
				# Seeded list to start examples with common Indian first names
				seed_first_names = [
					'Tharun','Varun','Arjun','Aarav','Vihaan','Aditya','Rohan','Rahul','Karthik','Siddharth',
					'Aryan','Ishaan','Kunal','Anirudh','Ritvik','Manish','Saurabh','Abhishek','Amit','Nikhil',
					'Neha','Priya','Ananya','Aishwarya','Pooja','Divya','Sneha','Ritika','Kavya','Soumya'
				]
				if i < len(seed_first_names):
					first = seed_first_names[i]
					last = random.choice(['Sharma','Verma','Reddy','Gupta','Patel','Iyer','Kumar','Singh','Nair','Das'])
					founder_name = f"{first} {last}"
				else:
					# Faker Indian locale fallback
					founder_name = fake_in.name()
			else:
				founder_name = fake.name()
			email = fake.email()
			role = random.choice(self.roles)
			company = self._generate_company_name()
			location = random.choice(self.locations)
			stage = random.choice(self.stages)
			
			# Industry and keywords
			primary_industry = random.choice(list(self.industry_keywords.keys()))
			secondary_industries = random.sample(
				[k for k in self.industry_keywords.keys() if k != primary_industry], 
				k=random.randint(0, 2)
			)
			
			all_industries = [primary_industry] + secondary_industries
			keywords = []
			
			for industry in all_industries:
				keywords.append(industry)
				keywords.extend(random.sample(
					self.industry_keywords[industry], 
					k=random.randint(1, 2)
				))
			
			# Add some general business keywords
			general_keywords = ['startup', 'growth', 'innovation', 'technology', 'digital transformation']
			keywords.extend(random.sample(general_keywords, k=random.randint(0, 2)))
			
			keywords_str = ', '.join(sorted(set(keywords))[:8])  # Limit and dedupe
			
			# Generate idea
			idea = self._generate_idea(primary_industry, keywords)
			
			# Generate about section
			about = self._generate_about(role, primary_industry)
			
			# LinkedIn (mix of real-looking URLs and placeholders)
			if random.random() < 0.7:
				linkedin_handle = founder_name.lower().replace(' ', '')
				linked_in = f"https://linkedin.com/in/{linkedin_handle}"
			else:
				linked_in = "Available upon request"
			
			# Optional notes
			notes = ""
			if random.random() < 0.3:
				notes_options = [
					"Open to co-founder discussions",
					"Actively fundraising",
					"Looking for technical advisor",
					"Interested in partnerships",
					"Seeking mentor connections",
					"Available for investor meetings"
				]
				notes = random.choice(notes_options)
			
			founder = {
				'id': founder_id,
				'founder_name': founder_name,
				'email': email,
				'role': role,
				'company': company,
				'location': location,
				'idea': idea,
				'about': about,
				'keywords': keywords_str,
				'stage': stage,
				'linked_in': linked_in,
				'notes': notes
			}
			
			founders.append(founder)
		
		print(f"Generated {len(founders)} founder profiles")
		return founders
	
	def _generate_company_name(self) -> str:
		"""Generate realistic company names"""
		prefixes = ['', 'Smart', 'Digital', 'Next', 'Future', 'Meta', 'Ultra', 'Pro', 'Elite', 'Prime']
		suffixes = ['Tech', 'Labs', 'Systems', 'Solutions', 'AI', 'Platform', 'Works', 'Hub', 'Cloud', '']
		
		base_words = [
			'Vision', 'Flow', 'Peak', 'Sync', 'Edge', 'Core', 'Wave', 'Spark', 'Shift', 'Scale',
			'Bridge', 'Link', 'Nexus', 'Orbit', 'Pulse', 'Quantum', 'Vertex', 'Matrix', 'Fusion'
		]
		
		if random.random() < 0.3:
			# Two-word combination
			return f"{random.choice(base_words)}{random.choice(base_words)}"
		else:
			# Prefix + base + suffix
			prefix = random.choice(prefixes)
			base = random.choice(base_words)
			suffix = random.choice(suffixes)
			
			parts = [p for p in [prefix, base, suffix] if p]
			return ''.join(parts)
	
	def _generate_idea(self, industry: str, keywords: List[str]) -> str:
		"""Generate realistic startup ideas"""
		industry_ideas = {
			'healthtech': [
				"AI-powered platform for remote patient monitoring and predictive health analytics",
				"Telemedicine marketplace connecting specialists with underserved communities",
				"Digital therapeutics app for mental health and wellness management"
			],
			'fintech': [
				"Embedded payments infrastructure for B2B marketplaces and platforms",
				"AI-driven personal finance app with automated savings and investment recommendations",
				"Blockchain-based supply chain financing for small businesses"
			],
			'ai': [
				"Computer vision platform for automated quality control in manufacturing",
				"NLP-powered customer service automation for enterprise clients",
				"Machine learning infrastructure for real-time fraud detection"
			],
			'saas': [
				"All-in-one workspace management platform for remote teams and hybrid work",
				"Customer success automation platform with predictive churn analysis",
				"No-code workflow builder for business process automation"
			],
			'ecommerce': [
				"Social commerce platform enabling creators to monetize their communities",
				"AI-powered inventory optimization and demand forecasting for retailers",
				"Sustainable marketplace connecting eco-conscious consumers with green brands"
			],
			'edtech': [
				"Adaptive learning platform using AI to personalize educational content",
				"Professional skills training marketplace with industry-verified certifications",
				"Virtual reality platform for immersive STEM education"
			]
		}
		
		default_ideas = [
			"Innovative platform connecting businesses with next-generation solutions",
			"Technology-driven marketplace solving real-world problems at scale",
			"Data analytics platform providing actionable insights for decision makers"
		]
		
		ideas = industry_ideas.get(industry, default_ideas)
		base_idea = random.choice(ideas)
		
		# Add some variation
		variations = [
			f"{base_idea}. Currently in beta with early enterprise customers.",
			f"{base_idea}. Focusing on the {random.choice(['European', 'Asian', 'Latin American', 'North American'])} market initially.",
			f"{base_idea}. Built by former {random.choice(['Google', 'Meta', 'Apple', 'Microsoft', 'Tesla'])} engineers."
		]
		
		return random.choice([base_idea] + variations)
	
	def _generate_about(self, role: str, industry: str) -> str:
		"""Generate realistic founder backgrounds"""
		prev_companies = [
			'Google', 'Meta', 'Apple', 'Microsoft', 'Amazon', 'Tesla', 'Uber', 'Airbnb', 'Stripe',
			'Palantir', 'Salesforce', 'Oracle', 'Adobe', 'Netflix', 'SpaceX', 'OpenAI', 'Coinbase'
		]
		
		prev_roles = {
			'Founder': ['Product Manager', 'Engineer', 'Consultant', 'VP'],
			'Co-founder': ['Senior Engineer', 'Product Lead', 'Designer', 'Data Scientist'],
			'Engineer': ['Software Engineer', 'Senior Developer', 'Tech Lead', 'Architect'],
			'PM': ['Product Manager', 'Strategy Consultant', 'Business Analyst', 'Operations'],
			'Investor': ['Investment Banker', 'Consultant', 'Analyst', 'Partner'],
			'Other': ['Consultant', 'Analyst', 'Manager', 'Director']
		}
		
		achievements = [
			f"Led team of {random.randint(5, 50)} engineers on high-impact projects",
			f"Shipped products used by {random.randint(1, 100)}M+ users worldwide",
			f"Raised ${random.randint(2, 50)}M in previous ventures",
			f"Generated ${random.randint(10, 500)}M+ in revenue at previous company",
			f"Built and scaled {random.randint(3, 20)}+ person team from scratch"
		]
		
		skills = [
			"Expert in full-stack development and cloud architecture",
			"Deep experience in machine learning and data science",
			"Strong background in product strategy and go-to-market",
			"Expertise in scaling B2B SaaS platforms",
			"Proven track record in international market expansion"
		]
		
		current_focus = [
			"Now focused on building the next generation of enterprise tools",
			"Passionate about solving complex problems with elegant technology",
			"Dedicated to creating products that make a meaningful impact",
			"Committed to building inclusive and diverse tech companies"
		]
		
		template = random.choice(self.background_templates)
		
		return template.format(
			prev_role=random.choice(prev_roles.get(role, ['Professional'])),
			prev_company=random.choice(prev_companies),
			years=random.randint(3, 15),
			industry=industry,
			achievement=random.choice(achievements),
			skills=random.choice(skills),
			specialty=f"{industry} and product development",
			current_focus=random.choice(current_focus),
			education=random.choice(['Stanford', 'MIT', 'Harvard', 'Berkeley', 'Carnegie Mellon']),
			current_role=role.lower(),
			passion=f"Passionate about {industry} innovation",
			team_type=random.choice(['engineering teams', 'product teams', 'data teams']),
			expertise=f"Deep expertise in {industry} and scaling technology",
			vision=f"Vision to transform the {industry} industry"
		)
	
	def create_searchable_text(self, founder: Dict) -> str:
		"""Create searchable text for embedding"""
		# Combine key fields with simple weights
		searchable_parts = []
		
		# High weight fields
		if founder.get('keywords'):
			searchable_parts.extend([founder['keywords']] * 3)
		
		if founder.get('role'):
			searchable_parts.extend([founder['role']] * 2)
		
		# Medium weight fields
		if founder.get('about'):
			searchable_parts.extend([founder['about']] * 2)
		
		if founder.get('idea'):
			searchable_parts.extend([founder['idea']] * 2)
		
		# Lower weight fields
		searchable_parts.extend([
			founder.get('company', ''),
			founder.get('location', ''),
			founder.get('stage', '')
		])
		
		return ' '.join([part for part in searchable_parts if part])
	
	def compute_embeddings(self, founders: List[Dict], api_token: str) -> Dict:
		"""Compute embeddings for all founders using Hugging Face API"""
		print("🔄 Computing embeddings...")
		
		embeddings_data = {}
		
		# Initialize Hugging Face embeddings helper
		hf_embeddings = HuggingFaceEmbeddings(api_token)
		status = hf_embeddings.get_status()
		
		if not status["hf_api_available"]:
			print("HF_API_TOKEN not provided or invalid. Writing searchable_text only (no vectors).")
			print("Will use TF-IDF fallback in the application.")
			for founder in founders:
				searchable_text = self.create_searchable_text(founder)
				embeddings_data[founder['id']] = {
					'searchable_text': searchable_text
				}
			return embeddings_data
		
		print(f"✅ Using Hugging Face API with model: {status['model_name']}")
		
		# Compute embeddings using Hugging Face API
		successful_embeddings = 0
		failed_embeddings = 0
		
		for i, founder in enumerate(founders):
			if i % 50 == 0:
				print(f"Computed embeddings for {i}/{len(founders)} founders...")
			
			searchable_text = self.create_searchable_text(founder)
			
			try:
				embedding = hf_embeddings.get_embedding(searchable_text)
				if embedding:
					embeddings_data[founder['id']] = {
						'embedding': embedding,
						'searchable_text': searchable_text
					}
					successful_embeddings += 1
				else:
					print(f"⚠️  Failed to get embedding for {founder['founder_name']}")
					# Store searchable_text for fallback
					embeddings_data[founder['id']] = {
						'searchable_text': searchable_text
					}
					failed_embeddings += 1
			except Exception as e:
				print(f"⚠️  Failed to compute embedding for {founder['founder_name']}: {e}")
				# Even on failure, store searchable_text for fallback
				embeddings_data[founder['id']] = {
					'searchable_text': searchable_text
				}
				failed_embeddings += 1
		
		print(f"✅ Embedding computation complete:")
		print(f"   • Successful embeddings: {successful_embeddings}")
		print(f"   • Failed embeddings: {failed_embeddings}")
		print(f"   • Total entries: {len(embeddings_data)}")
		
		return embeddings_data

def main():
	"""Main data generation script"""
	print("FounderMatch RAG Data Generation Script")
	print("=" * 50)
	
	parser = argparse.ArgumentParser(description="Generate founders CSV and embeddings JSON")
	parser.add_argument('--rows', type=int, default=700, help='Number of founder rows to generate')
	args = parser.parse_args()
	
	api_token = os.getenv('HF_API_TOKEN', '')
	if api_token:
		print("Using HF_API_TOKEN from environment for embeddings.")
	else:
		print("HF_API_TOKEN not set. Proceeding without embeddings (TF-IDF fallback).")
	
	# Initialize generator
	generator = DataGenerator()
	
	# Generate founder data
	founders = generator.generate_founder_data(args.rows)
	
	# Ensure data directory exists
	ensure_data_dir()
	
	# Save CSV
	csv_path = os.path.join(DATA_DIR, 'founders.csv')
	fieldnames = [
		'id', 'founder_name', 'email', 'role', 'company', 'location',
		'idea', 'about', 'keywords', 'stage', 'linked_in', 'notes'
	]
	
	print(f"Saving CSV to {csv_path}...")
	with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(founders)
	
	# Compute and save embeddings (or searchable_text only)
	embeddings_data = generator.compute_embeddings(founders, api_token)
	
	embeddings_path = os.path.join(DATA_DIR, 'embeddings.json')
	print(f"Saving embeddings to {embeddings_path}...")
	with open(embeddings_path, 'w', encoding='utf-8') as f:
		json.dump(embeddings_data, f, indent=2)
	
	print("=" * 50)
	print("Data generation complete!")
	print(f"   • Generated {len(founders)} founder profiles")
	print(f"   • Embedding entries written: {len(embeddings_data)}")
	print(f"   • Saved CSV: {csv_path}")
	print(f"   • Saved embeddings: {embeddings_path}")
	
	# Show sample data
	print("\nSample founder profiles:")
	for i, founder in enumerate(founders[:3]):
		print(f"\n{i+1}. {founder['founder_name']} ({founder['role']} at {founder['company']})")
		print(f"   Location: {founder['location']}")
		print(f"   Keywords: {founder['keywords']}")
		print(f"   Idea: {founder['idea'][:100]}...")

if __name__ == "__main__":
	main()