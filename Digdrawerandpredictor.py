from pathlib import Path
import numpy as np
import pygame 


def relu(z):
	return np.maximum(0, z)


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def load_weights(weights_file):
	data = np.load(weights_file)
	return data["W1"], data["b1"], data["W2"], data["b2"], data["W3"], data["b3"]


def find_weights_path():
	base_dir = Path(__file__).resolve().parent
	candidates = [
		base_dir / "mnist_weights.npz",
		base_dir.parent / "mnist_weights.npz",
		Path.cwd() / "mnist_weights.npz",
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(
		"Weights file not found. Checked: " + ", ".join(str(p) for p in candidates)
	)


def forward_pass(x, W1, b1, W2, b2, W3, b3):
	z1 = np.dot(W1, x) + b1
	a1 = relu(z1)
	z2 = np.dot(W2, a1) + b2
	a2 = relu(z2)
	z3 = np.dot(W3, a2) + b3
	a3 = sigmoid(z3)
	return a3


def predict_with_confidence(x_input, W1, b1, W2, b2, W3, b3):
	output = forward_pass(x_input, W1, b1, W2, b2, W3, b3)
	prediction = int(np.argmax(output))
	confidence = float(output[prediction, 0] * 100.0)
	return prediction, confidence, output


def preprocess_canvas(draw_surface):
	arr = pygame.surfarray.array3d(draw_surface)
	gray = np.mean(arr, axis=2)
	nonzero = np.argwhere(gray > 20)

	if nonzero.size == 0:
		return None

	x_min, y_min = np.min(nonzero[:, 0]), np.min(nonzero[:, 1])
	x_max, y_max = np.max(nonzero[:, 0]), np.max(nonzero[:, 1])
	width = int(x_max - x_min + 1)
	height = int(y_max - y_min + 1)

	crop_rect = pygame.Rect(int(x_min), int(y_min), width, height)
	cropped = draw_surface.subsurface(crop_rect).copy()

	scale = min(20.0 / max(width, 1), 20.0 / max(height, 1))
	new_w = max(1, int(round(width * scale)))
	new_h = max(1, int(round(height * scale)))
	resized = pygame.transform.smoothscale(cropped, (new_w, new_h))

	mnist_surface = pygame.Surface((28, 28))
	mnist_surface.fill((0, 0, 0))
	offset_x = (28 - new_w) // 2
	offset_y = (28 - new_h) // 2
	mnist_surface.blit(resized, (offset_x, offset_y))

	mnist_arr = pygame.surfarray.array3d(mnist_surface)
	mnist_gray = np.mean(mnist_arr, axis=2).T
	mnist_gray = mnist_gray.astype(np.float32) / 255.0
	x_input = mnist_gray.reshape(784, 1)
	return x_input


def main():
	weights_path = find_weights_path()

	W1, b1, W2, b2, W3, b3 = load_weights(weights_path)

	pygame.init()
	pygame.display.set_caption("Draw Digit Predictor")
	screen_w, screen_h = 760, 420
	draw_size = 280
	draw_pos = (20, 20)
	screen = pygame.display.set_mode((screen_w, screen_h))
	clock = pygame.time.Clock()

	font = pygame.font.SysFont("consolas", 24)
	small_font = pygame.font.SysFont("consolas", 18)

	canvas = pygame.Surface((draw_size, draw_size))
	canvas.fill((0, 0, 0))

	drawing = False
	brush_radius = 12
	prediction = None
	confidence = None

	running = True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

			if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				mx, my = event.pos
				if (
					draw_pos[0] <= mx < draw_pos[0] + draw_size
					and draw_pos[1] <= my < draw_pos[1] + draw_size
				):
					drawing = True

			if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
				drawing = False

			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_c:
					canvas.fill((0, 0, 0))
					prediction = None
					confidence = None

				if event.key == pygame.K_p:
					x_input = preprocess_canvas(canvas)
					if x_input is not None:
						prediction, confidence, _ = predict_with_confidence(
							x_input, W1, b1, W2, b2, W3, b3
						)
					else:
						prediction = None
						confidence = None

		if drawing:
			mx, my = pygame.mouse.get_pos()
			cx = mx - draw_pos[0]
			cy = my - draw_pos[1]
			if 0 <= cx < draw_size and 0 <= cy < draw_size:
				pygame.draw.circle(canvas, (255, 255, 255), (cx, cy), brush_radius)

		screen.fill((22, 24, 28))
		pygame.draw.rect(screen, (40, 43, 50), (10, 10, 300, 300), border_radius=8)
		screen.blit(canvas, draw_pos)

		title = font.render("Digit Drawer", True, (240, 240, 240))
		screen.blit(title, (340, 24))

		help_1 = small_font.render("Draw with mouse (left click + drag)", True, (210, 210, 210))
		help_2 = small_font.render("Press P to predict", True, (210, 210, 210))
		help_3 = small_font.render("Press C to clear", True, (210, 210, 210))
		screen.blit(help_1, (340, 80))
		screen.blit(help_2, (340, 112))
		screen.blit(help_3, (340, 144))

		if prediction is not None:
			pred_text = font.render(f"Prediction: {prediction}", True, (120, 255, 140))
			screen.blit(pred_text, (340, 210))
			if confidence is not None:
				conf_text = font.render(f"Confidence: {confidence:.1f}%", True, (120, 200, 255))
				screen.blit(conf_text, (340, 246))
		else:
			wait_text = small_font.render("Prediction will appear here", True, (180, 180, 180))
			screen.blit(wait_text, (340, 220))

		pygame.display.flip()
		clock.tick(120)

	pygame.quit()


if __name__ == "__main__":
	main()
