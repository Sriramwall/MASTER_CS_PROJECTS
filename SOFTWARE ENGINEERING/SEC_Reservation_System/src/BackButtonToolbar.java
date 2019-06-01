import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.*;

public class BackButtonToolbar extends JPanel implements ActionListener{

	private JButton nextButton;
		
	private ButtonListener listener; 
	
	public BackButtonToolbar() {
		nextButton = new JButton("Next");
		nextButton.addActionListener(this);
		setLayout(new FlowLayout(FlowLayout.CENTER));
		add(nextButton);
	}
	
	public void setButtonListener(ButtonListener buttonListner) {
		this.listener = buttonListner;
	}
	@Override
	public void actionPerformed(ActionEvent e) {
		JButton clicked = (JButton)e.getSource();
		
		if(clicked == nextButton) {
			if(listener != null) {
				listener.actionEmitted("next");
			}
		}		
	}
	
	
}
