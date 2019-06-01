import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.*;

public class ButtonToolbar extends JPanel implements ActionListener{

	private JButton presentButton;
	private JButton absentButton;
	
	private ButtonListener listener; 
	
	public ButtonToolbar() {
		presentButton = new JButton("Present");
		absentButton = new JButton("Absent");
		
		presentButton.addActionListener(this);
		absentButton.addActionListener(this);
		
		setLayout(new FlowLayout(FlowLayout.CENTER));
		
		add(presentButton);
		add(absentButton);
	}
	
	public void setButtonListener(ButtonListener buttonListner) {
		this.listener = buttonListner;
	}
	@Override
	public void actionPerformed(ActionEvent e) {
		JButton clicked = (JButton)e.getSource();
		
		if(clicked == presentButton) {
			if(listener != null) {
				listener.actionEmitted("Present");
			}
		}
		else if(clicked == absentButton) {
			if(listener != null) {
				listener.actionEmitted("Absent");
			}
		}
		
	}
	
	
}
