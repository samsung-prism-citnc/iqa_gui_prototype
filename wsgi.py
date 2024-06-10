from stable_diffusion import StableDiffusion

model = StableDiffusion()
image = model.generate_image('a cat in a forest')
image.show()