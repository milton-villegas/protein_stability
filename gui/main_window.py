"""
Main application window with tabs
Protein Stability DoE Suite
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from core.project import DoEProject
from utils.plotting import setup_plot_style


class ProteinDoESuite(tk.Tk):
    """Main application window"""

    def __init__(self):
        super().__init__()

        self.title("Protein Stability DoE Suite v1.0.0")
        self.geometry("1200x800")

        # Setup matplotlib style
        setup_plot_style()

        # Shared project data
        self.project = DoEProject()

        # Create UI
        self._create_menu()
        self._create_tabs()

        # Status bar
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="New Project", command=self._new_project, accelerator="Ctrl+N")
        file_menu.add_command(label="Open Project...", command=self._open_project, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Project", command=self._save_project, accelerator="Ctrl+S")
        file_menu.add_command(label="Save Project As...", command=self._save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export Design (Excel)...", command=self._export_design_excel)
        file_menu.add_command(label="Export Design (CSV)...", command=self._export_design_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Import Results (Excel)...", command=self._import_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

        # Keyboard shortcuts
        self.bind('<Control-n>', lambda e: self._new_project())
        self.bind('<Control-o>', lambda e: self._open_project())
        self.bind('<Control-s>', lambda e: self._save_project())

    def _create_tabs(self):
        """Create tab interface"""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Import tab classes from new module structure
        from gui.tabs.designer import DesignerTab
        from gui.tabs.analysis import AnalysisTab

        # Create tabs
        self.designer_tab = DesignerTab(self.notebook, self.project, self)
        self.analysis_tab = AnalysisTab(self.notebook, self.project, self)

        self.notebook.add(self.designer_tab, text="Design")
        self.notebook.add(self.analysis_tab, text="Analysis")

        # Create Results tab last (always visible, empty until analysis runs)
        self.analysis_tab.create_results_tab()

    # ========== Project Management ==========

    def _new_project(self):
        """Create new project"""
        if messagebox.askyesno("New Project", "Create new project? Unsaved changes will be lost."):
            self.project = DoEProject()
            self.designer_tab.refresh()
            self.analysis_tab.refresh()
            self.update_status("New project created")

    def _open_project(self):
        """Load project from file"""
        filepath = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("DoE Project", "*.doe"), ("All Files", "*.*")]
        )

        if filepath:
            try:
                self.project = DoEProject.load(filepath)
                self.designer_tab.project = self.project
                self.analysis_tab.project = self.project
                self.designer_tab.refresh()
                self.analysis_tab.refresh()
                self.update_status(f"Loaded: {filepath}")
            except Exception as e:
                messagebox.showerror("Project Load Failed",
                    f"Could not open the selected project file.\n\n"
                    f"Details: {str(e)}\n\n"
                    f"Make sure the file is a valid .doe project file.")

    def _save_project(self):
        """Save current project"""
        # Simple save (would need to track filepath)
        self._save_project_as()

    def _save_project_as(self):
        """Save project to new file"""
        filepath = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".doe",
            filetypes=[("DoE Project", "*.doe"), ("All Files", "*.*")]
        )

        if filepath:
            try:
                self.project.save(filepath)
                self.update_status(f"Saved: {filepath}")
            except Exception as e:
                messagebox.showerror("Project Save Failed",
                    f"Could not save the project file.\n\n"
                    f"Details: {str(e)}\n\n"
                    f"Check that you have write permissions for the selected location.")

    # ========== Export/Import ==========

    def _export_design_excel(self):
        """Export design matrix to Excel"""
        if self.project.design_matrix is None:
            messagebox.showwarning("No Design", "Generate a design first in the Design tab")
            return

        self.notebook.select(self.designer_tab)
        self.designer_tab.export_excel()

    def _export_design_csv(self):
        """Export design to Opentrons CSV"""
        if self.project.design_matrix is None:
            messagebox.showwarning("No Design", "Generate a design first in the Design tab")
            return

        self.notebook.select(self.designer_tab)
        self.designer_tab.export_csv()

    def _import_results(self):
        """Import experimental results"""
        self.notebook.select(self.analysis_tab)
        self.analysis_tab.load_results()

    # ========== Utilities ==========

    def update_status(self, message: str):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.update_idletasks()

    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Protein Stability DoE Suite v1.0.0\n\n"
            "Design of Experiments tool for protein stability studies\n\n"
            "Author: Milton F. Villegas\n"
            "Email: miltonfvillegas@gmail.com"
        )
